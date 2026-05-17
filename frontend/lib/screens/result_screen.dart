import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:pdf/pdf.dart';
import 'package:pdf/widgets.dart' as pw;
import 'package:printing/printing.dart';

import '../config/api_config.dart';
import '../constants/api_error_codes.dart';
import '../models/analyze_response.dart';
import '../models/report_metrics.dart';
import '../models/result_screen_args.dart';
import '../ui/medical_ui.dart';

const double _kResultImageMaxHeight = 300;

const String _kMedicalDisclaimerTitle = '보조 판독 안내';
const String _kMedicalDisclaimerBody =
    '본 결과는 확정 진단이 아닌, AI가 참고한 영역을 색으로 표시한 보조 시각화입니다.\n'
    '표시된 색 영역이 병변 위치를 직접 뜻하지 않을 수 있습니다.';

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
      errorBuilder: (_, _, _) => const Center(
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

/// Scaffold 하단 슬롯 전용 — Column 안에 넣지 않음(세로 무한 확장 방지).
Widget _buildResultBottomBar({
  required double maxContentWidth,
  required VoidCallback onUpload,
  required VoidCallback onHistory,
}) {
  return Material(
    color: MedicalTokens.surface,
    elevation: 4,
    surfaceTintColor: Colors.transparent,
    shadowColor: const Color(0x33000000),
    child: Column(
      mainAxisSize: MainAxisSize.min,
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Divider(height: 1, thickness: 1, color: MedicalTokens.border),
        SafeArea(
          top: false,
          minimum: EdgeInsets.zero,
          child: Padding(
            padding: const EdgeInsets.symmetric(
              horizontal: MedicalTokens.spaceMd,
              vertical: 8,
            ),
            child: LayoutBuilder(
              builder: (context, constraints) {
                final barWidth = constraints.maxWidth.isFinite
                    ? constraints.maxWidth.clamp(0.0, maxContentWidth)
                    : maxContentWidth;
                return Align(
                  alignment: Alignment.topCenter,
                  child: SizedBox(
                    width: barWidth,
                    height: 44,
                    child: Row(
                      children: [
                        Expanded(
                          child: OutlinedButton(
                            style: OutlinedButton.styleFrom(
                              foregroundColor: MedicalTokens.textMain,
                              backgroundColor: MedicalTokens.surface,
                              side: const BorderSide(
                                color: MedicalTokens.border,
                              ),
                              minimumSize: const Size.fromHeight(44),
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 10,
                              ),
                            ),
                            onPressed: onUpload,
                            child: const Text('다시 업로드'),
                          ),
                        ),
                        const SizedBox(width: MedicalTokens.spaceSm),
                        Expanded(
                          child: FilledButton(
                            style: FilledButton.styleFrom(
                              backgroundColor: MedicalTokens.primary,
                              foregroundColor: Colors.white,
                              minimumSize: const Size.fromHeight(44),
                              padding: const EdgeInsets.symmetric(
                                horizontal: 16,
                                vertical: 10,
                              ),
                            ),
                            onPressed: onHistory,
                            child: const Text('이력 보기'),
                          ),
                        ),
                      ],
                    ),
                  ),
                );
              },
            ),
          ),
        ),
      ],
    ),
  );
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

    final screenW = MediaQuery.sizeOf(context).width;
    final barMaxWidth = screenW >= 920 ? 920.0 : 680.0;

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
                Expanded(
                  child: SingleChildScrollView(
                    padding: const EdgeInsets.all(MedicalTokens.spaceMd),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: BoxConstraints(maxWidth: contentWidth),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.stretch,
                          children: [
                            const MedicalNoticeBanner(
                              title: _kMedicalDisclaimerTitle,
                              body: _kMedicalDisclaimerBody,
                            ),
                            const SizedBox(height: MedicalTokens.spaceMd),
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
                            const MedicalSectionTitle('이상 확률'),
                            const SizedBox(height: MedicalTokens.spaceSm),
                            _ProbabilityCard(response: res),
                            const SizedBox(height: MedicalTokens.spaceLg),
                            _ReportMetricsSection(response: res),
                            const SizedBox(height: MedicalTokens.spaceMd),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            );
          },
        ),
      ),
      bottomNavigationBar: _buildResultBottomBar(
        maxContentWidth: barMaxWidth,
        onUpload: () {
          Navigator.pushNamedAndRemoveUntil(
            context,
            '/upload',
            (route) => false,
          );
        },
        onHistory: () {
          Navigator.pushNamed(context, '/history');
        },
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
        _pdfNoticeBanner(
          title: _kMedicalDisclaimerTitle,
          body: _kMedicalDisclaimerBody,
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
          '이상 확률',
          style: pw.TextStyle(fontSize: 15, fontWeight: pw.FontWeight.bold),
        ),
        pw.SizedBox(height: 8),
        _pdfInfoCard(pw.Text(_buildProbabilityPdf(response))),
        if (response?.evalMetrics != null) ...[
          pw.SizedBox(height: 16),
          pw.Text(
            '성능 지표',
            style: pw.TextStyle(fontSize: 15, fontWeight: pw.FontWeight.bold),
          ),
          pw.SizedBox(height: 8),
          _pdfInfoCard(_buildMetricsPdf(response)),
        ],
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

pw.Widget _pdfNoticeBanner({String? title, required String body}) {
  return pw.Container(
    width: double.infinity,
    padding: const pw.EdgeInsets.fromLTRB(12, 10, 12, 10),
    decoration: pw.BoxDecoration(
      color: PdfColor.fromHex('#E8F4FA'),
      border: pw.Border(
        left: pw.BorderSide(color: PdfColor.fromHex('#72A9C7'), width: 4),
        top: pw.BorderSide(color: PdfColor.fromHex('#D6E2EB')),
        right: pw.BorderSide(color: PdfColor.fromHex('#D6E2EB')),
        bottom: pw.BorderSide(color: PdfColor.fromHex('#D6E2EB')),
      ),
      borderRadius: pw.BorderRadius.circular(6),
    ),
    child: pw.Column(
      crossAxisAlignment: pw.CrossAxisAlignment.start,
      children: [
        if (title != null && title.isNotEmpty) ...[
          pw.Text(
            title,
            style: pw.TextStyle(fontSize: 11, fontWeight: pw.FontWeight.bold),
          ),
          pw.SizedBox(height: 4),
        ],
        pw.Text(body, style: const pw.TextStyle(fontSize: 9.5, lineSpacing: 4)),
      ],
    ),
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

  final metrics = response.evalMetrics;
  if (metrics == null) {
    return pw.Text('모델 평가 지표가 없어 리포트 지표를 표시할 수 없습니다.');
  }

  final rows = metrics.displayRows;
  return pw.Column(
    crossAxisAlignment: pw.CrossAxisAlignment.start,
    children: [
      for (var i = 0; i < rows.length; i++) ...[
        if (i > 0) pw.SizedBox(height: 8),
        pw.Text(
          '${rows[i].title}: ${ReportMetrics.formatPercent(rows[i].ratio)}',
        ),
        pw.SizedBox(height: 2),
        pw.Text(rows[i].subtitle, style: const pw.TextStyle(fontSize: 10)),
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

/// 성능 지표 블록 — eval_metrics 없으면 기본 숨김, 탭하면 안내만 펼침.
class _ReportMetricsSection extends StatefulWidget {
  const _ReportMetricsSection({required this.response});

  final AnalyzeResponse? response;

  @override
  State<_ReportMetricsSection> createState() => _ReportMetricsSectionState();
}

class _ReportMetricsSectionState extends State<_ReportMetricsSection> {
  bool _expanded = false;

  @override
  Widget build(BuildContext context) {
    final res = widget.response;
    if (res == null || !res.canShowInferenceResults) {
      return const SizedBox.shrink();
    }

    final metrics = res.evalMetrics;
    if (metrics != null) {
      return Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const MedicalSectionTitle(
            '성능 지표',
            subtitle: 'external test 기준 모델 성능 요약',
          ),
          const SizedBox(height: MedicalTokens.spaceSm),
          _ReportMetricsCard(response: res),
          const SizedBox(height: MedicalTokens.spaceLg),
        ],
      );
    }

    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InkWell(
          onTap: () => setState(() => _expanded = !_expanded),
          borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              children: [
                Expanded(
                  child: Text(
                    '성능 지표',
                    style: theme.textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: MedicalTokens.textSubtle,
                    ),
                  ),
                ),
                Text(
                  _expanded ? '접기' : '평가 데이터 없음',
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: MedicalTokens.textSubtle,
                  ),
                ),
                const SizedBox(width: 4),
                Icon(
                  _expanded
                      ? Icons.expand_less
                      : Icons.expand_more,
                  size: 20,
                  color: MedicalTokens.textSubtle,
                ),
              ],
            ),
          ),
        ),
        if (_expanded) ...[
          const SizedBox(height: MedicalTokens.spaceSm),
          const MedicalNoticeBanner(
            title: '성능 지표',
            body: '모델 평가 지표(JSON)가 아직 없어 성능 수치를 표시하지 않습니다.\n'
                'AI 팀에서 base.yaml 버전에 맞는 '
                'external_test_*_best_metrics.json을 제공하면 표시됩니다.',
          ),
        ],
        const SizedBox(height: MedicalTokens.spaceLg),
      ],
    );
  }
}

/// 백엔드 `/analyze` · 이력 `metrics` (AI external test eval_metrics).
class _ReportMetricsCard extends StatelessWidget {
  const _ReportMetricsCard({required this.response});

  final AnalyzeResponse? response;

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

    final metrics = res.evalMetrics;
    if (metrics == null) {
      return const SizedBox.shrink();
    }

    final rows = metrics.displayRows;
    final theme = Theme.of(context);
    return _InfoCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < rows.length; i++) ...[
            if (i > 0) const SizedBox(height: 12),
            Row(
              children: [
                Expanded(
                  child: Text(
                    rows[i].title,
                    style: theme.textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.w700,
                      color: MedicalTokens.textMain,
                    ),
                  ),
                ),
                MedicalBadge(
                  text: ReportMetrics.formatPercent(rows[i].ratio),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Text(
              rows[i].subtitle,
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

