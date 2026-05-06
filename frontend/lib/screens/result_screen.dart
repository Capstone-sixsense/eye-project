import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../config/api_config.dart';
import '../constants/api_error_codes.dart';
import '../models/analyze_response.dart';
import '../models/result_screen_args.dart';
import '../ui/medical_ui.dart';

const double _kResultImageMaxHeight = 300;

String? _originalAssetAbsoluteUrl(AnalyzeResponse? res) {
  final path = res?.originalUrl;
  if (path == null || path.isEmpty) return null;
  return ApiConfig.resolveAssetUrl(path);
}

Widget _buildOriginalImageFit({
  required Uint8List? storedBytes,
  required AnalyzeResponse? response,
}) {
  if (storedBytes != null) {
    return Image.memory(storedBytes, fit: BoxFit.contain);
  }
  final u = _originalAssetAbsoluteUrl(response);
  if (u != null && u.isNotEmpty) {
    return Image.network(
      u,
      fit: BoxFit.contain,
      loadingBuilder: (context, child, progress) {
        if (progress == null) return child;
        return const Center(child: CircularProgressIndicator());
      },
      errorBuilder: (_, __, ___) => const Center(
        child: Padding(
          padding: EdgeInsets.all(16),
          child: Text(
            '원본 이미지를 불러올 수 없습니다.',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
  return const Center(child: Text('이미지가 없습니다'));
}

class ResultScreen extends StatelessWidget {
  const ResultScreen({
    super.key,
    this.args,
    this.originalImageBytes,
  });

  final ResultScreenArgs? args;

  /// 이전 라우트 호환(원본만).
  final Uint8List? originalImageBytes;

  @override
  Widget build(BuildContext context) {
    final Uint8List? original = args?.originalImageBytes ?? originalImageBytes;
    final AnalyzeResponse? res = args?.analyzeResponse;

    final String? explanationPath = res?.resolvedExplanationPath;
    final String? explanationAbsoluteUrl =
        (explanationPath != null && explanationPath.isNotEmpty)
            ? ApiConfig.resolveAssetUrl(explanationPath)
            : null;
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: IconButton(
          onPressed: () {
            _showZoomViewer(
              context,
              storedBytes: original,
              response: res,
              explanationAbsoluteUrl: explanationAbsoluteUrl,
            );
          },
          style: IconButton.styleFrom(
            foregroundColor: Colors.black,
            backgroundColor: Colors.white,
            side: const BorderSide(color: Colors.black12),
          ),
          icon: const _MagnifierGlyph(size: 22),
          tooltip: '이미지 확대 보기',
        ),
        actions: [
          TextButton(
            onPressed: () => _exportResultPdf(
              context,
              original: original,
              response: res,
              explanationAbsoluteUrl: explanationAbsoluteUrl,
            ),
            child: const Text('내보내기'),
          ),
        ],
      ),
      body: SafeArea(
        child: LayoutBuilder(
          builder: (context, constraints) {
            final wide = constraints.maxWidth >= 920;
            final contentWidth = wide ? 920.0 : 680.0;

            return Column(
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                const Padding(
                  padding: EdgeInsets.fromLTRB(
                    MedicalTokens.spaceMd,
                    MedicalTokens.spaceMd,
                    MedicalTokens.spaceMd,
                    0,
                  ),
                  child: _MedicalDisclaimerBanner(),
                ),
                const SizedBox(height: MedicalTokens.spaceSm),
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(MedicalTokens.spaceMd),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: BoxConstraints(maxWidth: contentWidth),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            if (wide)
                              Row(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.stretch,
                                      children: [
                                        Text(
                                          '원본이미지',
                                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                                fontWeight: FontWeight.w700,
                                              ),
                                        ),
                                        const SizedBox(height: 8),
                                        _ImageBox(
                                          child: _buildOriginalImageFit(
                                            storedBytes: original,
                                            response: res,
                                          ),
                                        ),
                                      ],
                                    ),
                                  ),
                                  const SizedBox(width: MedicalTokens.spaceSm),
                                  Expanded(
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.stretch,
                                      children: [
                                        Text(
                                          '결과 이미지',
                                          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                                fontWeight: FontWeight.w700,
                                              ),
                                        ),
                                        const SizedBox(height: 8),
                                        _ExplanationPanel(
                                          absoluteUrl: explanationAbsoluteUrl,
                                          response: res,
                                        ),
                                      ],
                                    ),
                                  ),
                                ],
                              )
                            else ...[
                              Text(
                                '원본이미지',
                                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                      fontWeight: FontWeight.w700,
                                    ),
                              ),
                              const SizedBox(height: 8),
                              _ImageBox(
                                child: _buildOriginalImageFit(
                                  storedBytes: original,
                                  response: res,
                                ),
                              ),
                              const SizedBox(height: MedicalTokens.spaceMd),
                              Text(
                                '결과 이미지',
                                style: Theme.of(context).textTheme.titleSmall?.copyWith(
                                      fontWeight: FontWeight.w700,
                                    ),
                              ),
                              const SizedBox(height: 8),
                              _ExplanationPanel(
                                absoluteUrl: explanationAbsoluteUrl,
                                response: res,
                              ),
                            ],
                            const SizedBox(height: MedicalTokens.spaceLg),
                            const MedicalSectionTitle(
                              '분석 요약',
                              subtitle: '핵심 판정 결과를 먼저 확인하세요.',
                            ),
                            const SizedBox(height: MedicalTokens.spaceSm),
                            _JudgmentCard(response: res),
                            const SizedBox(height: MedicalTokens.spaceLg),
                            const MedicalSectionTitle('성능 지표'),
                            const SizedBox(height: MedicalTokens.spaceSm),
                            _ReportMetricsCard(response: res),
                            const SizedBox(height: MedicalTokens.spaceLg),
                            const MedicalSectionTitle('이상 확률'),
                            const SizedBox(height: MedicalTokens.spaceSm),
                            _ProbabilityCard(response: res),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
                Padding(
                  padding: const EdgeInsets.fromLTRB(
                    MedicalTokens.spaceMd,
                    8,
                    MedicalTokens.spaceMd,
                    MedicalTokens.spaceMd,
                  ),
                  child: Row(
                    children: [
                      Expanded(
                        child: MedicalSecondaryButton(
                          label: '다시 업로드',
                          onPressed: () {
                            Navigator.pushNamedAndRemoveUntil(
                              context,
                              '/upload',
                              (route) => false,
                            );
                          },
                        ),
                      ),
                      const SizedBox(width: MedicalTokens.spaceSm),
                      Expanded(
                        child: MedicalPrimaryButton(
                          label: '이력 보기',
                          onPressed: () {
                            Navigator.pushNamed(context, '/history');
                          },
                        ),
                      ),
                    ],
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

Future<void> _exportResultPdf(
  BuildContext context, {
  required Uint8List? original,
  required AnalyzeResponse? response,
  required String? explanationAbsoluteUrl,
}) async {
  showDialog<void>(
    context: context,
    barrierDismissible: false,
    builder: (_) => const Center(child: CircularProgressIndicator()),
  );

  try {
    Uint8List? mergedOriginal = original;
    if (mergedOriginal == null && response != null) {
      final u = _originalAssetAbsoluteUrl(response);
      mergedOriginal =
          u != null && u.isNotEmpty ? await _tryFetchImageBytes(u) : null;
    }

    final bytes = await _buildResultPdf(
      original: mergedOriginal,
      response: response,
      explanationAbsoluteUrl: explanationAbsoluteUrl,
    );

    if (context.mounted) {
      Navigator.of(context, rootNavigator: true).pop();
    }

    final now = DateTime.now();
    final filename =
        'result_report_${now.year}${now.month.toString().padLeft(2, '0')}${now.day.toString().padLeft(2, '0')}.pdf';

    await Printing.sharePdf(bytes: bytes, filename: filename);

    if (context.mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('PDF 내보내기가 완료되었습니다.')),
      );
    }
  } catch (_) {
    if (context.mounted) {
      Navigator.of(context, rootNavigator: true).pop();
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('PDF 내보내기에 실패했습니다. 다시 시도해주세요.')),
      );
    }
  }
}

Future<Uint8List> _buildResultPdf({
  required Uint8List? original,
  required AnalyzeResponse? response,
  required String? explanationAbsoluteUrl,
}) async {
  final doc = pw.Document();

  final fontRegular = await PdfGoogleFonts.notoSansKRRegular();
  final fontBold = await PdfGoogleFonts.notoSansKRBold();

  final explanationBytes = await _tryFetchImageBytes(explanationAbsoluteUrl);

  doc.addPage(
    pw.MultiPage(
      pageFormat: PdfPageFormat.a4,
      margin: const pw.EdgeInsets.all(24),
      theme: pw.ThemeData.withFont(base: fontRegular, bold: fontBold),
      build: (_) => [
        pw.Text(
          '결과 보고서',
          style: pw.TextStyle(fontSize: 20, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 10),
        pw.Container(
          width: double.infinity,
          padding: const pw.EdgeInsets.symmetric(horizontal: 10, vertical: 8),
          decoration: pw.BoxDecoration(
            color: PdfColor.fromHex('#FFF4E8'),
            border: pw.Border.all(color: PdfColor.fromHex('#F3D2AE')),
            borderRadius: pw.BorderRadius.circular(6),
          ),
          child: pw.Text('본 결과는 의료적 확정 진단이 아닌 보조 판별 결과입니다.'),
        ),
        pw.SizedBox(height: 16),
        pw.Text(
          '원본이미지',
          style: pw.TextStyle(fontSize: 13, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfImageBox(
          original != null
              ? pw.Image(pw.MemoryImage(original), fit: pw.BoxFit.contain)
              : pw.Center(child: pw.Text('이미지가 없습니다')),
        ),
        pw.SizedBox(height: 14),
        pw.Text(
          '결과 이미지',
          style: pw.TextStyle(fontSize: 13, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfImageBox(
          explanationBytes != null
              ? pw.Image(
                  pw.MemoryImage(explanationBytes),
                  fit: pw.BoxFit.contain,
                )
              : pw.Center(child: pw.Text('결과 이미지를 불러올 수 없습니다.')),
        ),
        pw.SizedBox(height: 18),
        pw.Text(
          '분석 요약',
          style: pw.TextStyle(fontSize: 15, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfInfoCard(_buildJudgmentPdf(response)),
        pw.SizedBox(height: 16),
        pw.Text(
          '성능 지표',
          style: pw.TextStyle(fontSize: 15, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfInfoCard(_buildMetricsPdf(response)),
        pw.SizedBox(height: 16),
        pw.Text(
          '이상 확률',
          style: pw.TextStyle(fontSize: 15, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfInfoCard(pw.Text(_buildProbabilityPdf(response))),
      ],
    ),
  );

  return doc.save();
}

Future<Uint8List?> _tryFetchImageBytes(String? url) async {
  if (url == null || url.isEmpty) return null;
  try {
    final uri = Uri.tryParse(url);
    if (uri == null) return null;
    final res = await http.get(uri);
    if (res.statusCode >= 200 && res.statusCode < 300) {
      return res.bodyBytes;
    }
  } catch (_) {
    return null;
  }
  return null;
}

pw.Widget _pdfImageBox(pw.Widget child) {
  return pw.Container(
    width: double.infinity,
    height: 180,
    padding: const pw.EdgeInsets.all(8),
    decoration: pw.BoxDecoration(
      border: pw.Border.all(color: PdfColor.fromHex('#D8DEE9')),
      borderRadius: pw.BorderRadius.circular(8),
    ),
    child: child,
  );
}

pw.Widget _pdfInfoCard(pw.Widget child) {
  return pw.Container(
    width: double.infinity,
    padding: const pw.EdgeInsets.all(12),
    decoration: pw.BoxDecoration(
      border: pw.Border.all(color: PdfColor.fromHex('#E6EAF2')),
      borderRadius: pw.BorderRadius.circular(8),
    ),
    child: child,
  );
}

pw.Widget _buildJudgmentPdf(AnalyzeResponse? response) {
  if (response == null) return pw.Text('결과가 없습니다.');
  if (response.isFail) return pw.Text(response.message ?? '분석에 실패했습니다.');
  if (!response.canShowInferenceResults) {
    return pw.Text('전처리를 통과하지 않아 판정 결과를 표시할 수 없습니다.');
  }
  return pw.Text('AI 판정: ${response.label ?? '—'}');
}

pw.Widget _buildMetricsPdf(AnalyzeResponse? response) {
  if (response == null) return pw.Text('—');
  if (response.isFail || !response.canShowInferenceResults) {
    return pw.Text('전처리를 통과하지 않아 리포트 지표를 표시할 수 없습니다.');
  }

  final prob = response.abnormalProbability;
  if (prob == null) {
    return pw.Text('이상 확률 값이 없어 리포트 지표를 계산할 수 없습니다.');
  }

  const labels = [
    ('Accuracy', '전체 성능 판단'),
    ('Precision', '불필요 오진 최소화'),
    ('Sensitivity', '놓치는 환자 최소화'),
    ('Specificity', '정상 오진 방지'),
    ('F1-score', '정밀도와 재현율 조화'),
  ];

  double metricValue(int index) => index == 3 ? 1.0 - prob : prob;

  return pw.Column(
    crossAxisAlignment: pw.CrossAxisAlignment.start,
    children: [
      for (var i = 0; i < labels.length; i++) ...[
        if (i > 0) pw.SizedBox(height: 8),
        pw.Text('${labels[i].$1}: ${(metricValue(i) * 100).toStringAsFixed(1)}%'),
        pw.SizedBox(height: 2),
        pw.Text(labels[i].$2, style: const pw.TextStyle(fontSize: 10)),
      ],
    ],
  );
}

String _buildProbabilityPdf(AnalyzeResponse? response) {
  if (response == null) return '—';
  if (response.isFail || !response.canShowInferenceResults) return '—';
  final prob = response.abnormalProbability;
  return prob == null ? '—' : '${(prob * 100).toStringAsFixed(1)}%';
}

void _showZoomViewer(
  BuildContext context, {
  required Uint8List? storedBytes,
  required AnalyzeResponse? response,
  required String? explanationAbsoluteUrl,
}) {
  showDialog<void>(
    context: context,
    builder: (dialogContext) {
      return Dialog.fullscreen(
        child: Scaffold(
          appBar: AppBar(
            automaticallyImplyLeading: false,
            centerTitle: true,
            title: IconButton(
              onPressed: () => Navigator.pop(dialogContext),
              icon: const Icon(Icons.close),
              tooltip: '닫기',
            ),
          ),
          body: Padding(
            padding: const EdgeInsets.all(MedicalTokens.spaceMd),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final isWide = constraints.maxWidth >= 920;
                      final originalPanel = _ZoomPanel(
                        title: '원본이미지',
                        child: _buildOriginalImageFit(
                          storedBytes: storedBytes,
                          response: response,
                        ),
                      );
                final explanationPanel = _ZoomPanel(
                  title: '결과 이미지',
                  child: (explanationAbsoluteUrl != null &&
                          explanationAbsoluteUrl.isNotEmpty)
                      ? Image.network(
                          explanationAbsoluteUrl,
                          fit: BoxFit.contain,
                          loadingBuilder: (context, child, progress) {
                            if (progress == null) return child;
                            return const Center(
                              child: CircularProgressIndicator(),
                            );
                          },
                          errorBuilder: (context, error, stackTrace) =>
                              const Center(
                            child: Padding(
                              padding: EdgeInsets.all(16),
                              child: Text(
                                '이미지를 불러올 수 없습니다.\n(CORS 또는 URL 확인)',
                                textAlign: TextAlign.center,
                              ),
                            ),
                          ),
                        )
                      : const Center(child: Text('응답에 결과 이미지 URL이 없습니다.')),
                );

                if (isWide) {
                  return Row(
                    children: [
                      Expanded(child: originalPanel),
                      const SizedBox(width: MedicalTokens.spaceMd),
                      Expanded(child: explanationPanel),
                    ],
                  );
                }

                return Column(
                  children: [
                    Expanded(child: originalPanel),
                    const SizedBox(height: MedicalTokens.spaceMd),
                    Expanded(child: explanationPanel),
                  ],
                );
              },
            ),
          ),
        ),
      );
    },
  );
}

class _ImageBox extends StatelessWidget {
  const _ImageBox({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return ConstrainedBox(
      constraints: const BoxConstraints(maxHeight: _kResultImageMaxHeight),
      child: MedicalCard(
        padding: EdgeInsets.zero,
        child: ClipRRect(
          borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
          child: SizedBox(
            height: _kResultImageMaxHeight,
            child: ColoredBox(
              color: Colors.black,
              child: child,
            ),
          ),
        ),
      ),
    );
  }
}

class _ZoomPanel extends StatelessWidget {
  const _ZoomPanel({
    required this.title,
    required this.child,
  });

  final String title;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          title,
          style: Theme.of(context).textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w700,
              ),
        ),
        const SizedBox(height: 8),
        Expanded(
          child: MedicalCard(
            padding: EdgeInsets.zero,
            child: ClipRRect(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
              child: InteractiveViewer(
                minScale: 1,
                maxScale: 5,
                child: ColoredBox(
                  color: Colors.black,
                  child: Center(child: child),
                ),
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _MagnifierGlyph extends StatelessWidget {
  const _MagnifierGlyph({required this.size});

  final double size;

  @override
  Widget build(BuildContext context) {
    final lensSize = size * 0.62;
    final handleWidth = size * 0.46;
    final handleThickness = size * 0.14;

    return SizedBox(
      width: size,
      height: size,
      child: Stack(
        clipBehavior: Clip.none,
        children: [
          Positioned(
            left: 1,
            top: 1,
            child: Container(
              width: lensSize,
              height: lensSize,
              decoration: BoxDecoration(
                shape: BoxShape.circle,
                border: Border.all(
                  color: Colors.black,
                  width: 2.2,
                ),
              ),
            ),
          ),
          Positioned(
            right: 0,
            bottom: 1,
            child: Transform.rotate(
              angle: 0.78,
              child: Container(
                width: handleWidth,
                height: handleThickness,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(999),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _JudgmentCard extends StatelessWidget {
  const _JudgmentCard({required this.response});

  final AnalyzeResponse? response;

  @override
  Widget build(BuildContext context) {
    final res = response;
    if (res == null) {
      return const _InfoCard(child: Text('결과가 없습니다.'));
    }
    if (res.isFail) {
      return _InfoCard(
        child: Text(res.message ?? '분석에 실패했습니다.'),
      );
    }
    if (!res.canShowInferenceResults) {
      return const _InfoCard(
        child: Text(
          '전처리를 통과하지 않아 판정 결과를 표시할 수 없습니다.',
        ),
      );
    }
    return _InfoCard(
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('AI 판정'),
                const SizedBox(height: 6),
                Text(
                  res.label ?? '—',
                  style: Theme.of(context).textTheme.headlineSmall,
                ),
              ],
            ),
          ),
          MedicalBadge(
            text: '완료',
            backgroundColor: const Color(0xFFE8F7F2),
            foregroundColor: MedicalTokens.success,
          ),
        ],
      ),
    );
  }
}

/// 백엔드 `make_result_img` / `analyze` 의 metrics dict와 동일한 5지표.
class _ReportMetricsCard extends StatelessWidget {
  const _ReportMetricsCard({required this.response});

  final AnalyzeResponse? response;

  static const List<(String, String)> _labels = [
    ('Accuracy', '전체 성능 판단'),
    ('Precision', '불필요 오진 최소화'),
    ('Sensitivity', '놓치는 환자 최소화'),
    ('Specificity', '정상 오진 방지'),
    ('F1-score', '정밀도와 재현율 조화'),
  ];

  static double _metricValue(int index, double prob) {
    if (index == 3) {
      return 1.0 - prob;
    }
    return prob;
  }

  @override
  Widget build(BuildContext context) {
    final res = response;
    if (res == null) {
      return const _InfoCard(child: Text('—'));
    }
    if (res.isFail || !res.canShowInferenceResults) {
      return const _InfoCard(
        child: Text(
          '전처리를 통과하지 않아 리포트 지표를 표시할 수 없습니다.',
        ),
      );
    }

    final prob = res.abnormalProbability;
    if (prob == null) {
      return const _InfoCard(
        child: Text('이상 확률 값이 없어 리포트 지표를 계산할 수 없습니다.'),
      );
    }

    final theme = Theme.of(context);
    return _InfoCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < _labels.length; i++) ...[
            if (i > 0) const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: Text(
                    _labels[i].$1,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: MedicalTokens.textMain,
                    ),
                  ),
                ),
                MedicalBadge(
                  text: '${(_metricValue(i, prob) * 100).toStringAsFixed(1)}%',
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              _labels[i].$2,
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _ProbabilityCard extends StatelessWidget {
  const _ProbabilityCard({required this.response});

  final AnalyzeResponse? response;

  @override
  Widget build(BuildContext context) {
    final res = response;
    if (res == null) {
      return const _InfoCard(child: Text('—'));
    }
    if (res.isFail || !res.canShowInferenceResults) {
      return const _InfoCard(child: Text('—'));
    }

    final prob = res.abnormalProbability;

    return _InfoCard(
      child: Row(
        children: [
          const Expanded(child: Text('이상 확률')),
          MedicalBadge(
            text: prob != null ? '${(prob * 100).toStringAsFixed(1)}%' : '—',
            backgroundColor: const Color(0xFFFFF4E8),
            foregroundColor: const Color(0xFFC0702D),
          ),
        ],
      ),
    );
  }
}

class _ExplanationPanel extends StatelessWidget {
  const _ExplanationPanel({
    required this.absoluteUrl,
    required this.response,
  });

  final String? absoluteUrl;
  final AnalyzeResponse? response;

  @override
  Widget build(BuildContext context) {
    final res = response;

    if (res != null && res.shouldShowExplanationFailure) {
      final code =
          res.xaiErrorCode ?? ApiErrorCodes.xaiGenerationFailed;
      return _InfoCard(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              '결과 이미지 생성에 실패했습니다',
              style: Theme.of(context).textTheme.bodyLarge,
            ),
            const SizedBox(height: 24),
            Text(
              code,
              style: Theme.of(context).textTheme.labelLarge?.copyWith(
                    color: Theme.of(context).colorScheme.error,
                  ),
            ),
          ],
        ),
      );
    }

    if (absoluteUrl != null && absoluteUrl!.isNotEmpty) {
      return ConstrainedBox(
        constraints: const BoxConstraints(maxHeight: _kResultImageMaxHeight),
        child: MedicalCard(
          padding: EdgeInsets.zero,
          child: ClipRRect(
            borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
            child: SizedBox(
              height: _kResultImageMaxHeight,
              child: ColoredBox(
                color: Colors.black,
                child: Image.network(
                  absoluteUrl!,
                  fit: BoxFit.contain,
                  loadingBuilder: (context, child, progress) {
                    if (progress == null) return child;
                    return const Center(child: CircularProgressIndicator());
                  },
                  errorBuilder: (context, error, stackTrace) => const Center(
                    child: Padding(
                      padding: EdgeInsets.all(16),
                      child: Text(
                        '이미지를 불러올 수 없습니다.\n(CORS 또는 URL 확인)',
                        textAlign: TextAlign.center,
                      ),
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      );
    }

    return const _InfoCard(
      child: Text('응답에 결과 이미지 URL이 없습니다.'),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return MedicalCard(child: child);
  }
}

class _MedicalDisclaimerBanner extends StatelessWidget {
  const _MedicalDisclaimerBanner();

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFFFFF4E8),
        borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
        border: Border.all(color: const Color(0xFFF3D2AE)),
      ),
      child: const Padding(
        padding: EdgeInsets.symmetric(
          horizontal: MedicalTokens.spaceSm,
          vertical: 10,
        ),
        child: Text(
          '본 결과는 의료적 확정 진단이 아닌 보조 판별 결과입니다.',
          textAlign: TextAlign.center,
          style: TextStyle(
            fontWeight: FontWeight.w700,
            color: Color(0xFF9A5E20),
          ),
        ),
      ),
    );
  }
}
