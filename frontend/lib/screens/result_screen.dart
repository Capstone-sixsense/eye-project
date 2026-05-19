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

/// 상단 바 — 하단 바와 동일한 surface·하단 구분선·그림자.
PreferredSizeWidget _buildMedicalTopBar({required Widget toolbar}) {
  return PreferredSize(
    preferredSize: const Size.fromHeight(kToolbarHeight + 1),
    child: Material(
      color: MedicalTokens.surface,
      elevation: 4,
      surfaceTintColor: Colors.transparent,
      shadowColor: const Color(0x33000000),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        children: [
          SafeArea(
            bottom: false,
            child: SizedBox(
              height: kToolbarHeight,
              child: toolbar,
            ),
          ),
          const Divider(
            height: 1,
            thickness: 1,
            color: MedicalTokens.border,
          ),
        ],
      ),
    ),
  );
}

PreferredSizeWidget _buildZoomViewerTopBar({
  required VoidCallback onClose,
  required bool drawMode,
  required VoidCallback onToggleDraw,
  required VoidCallback onClearAll,
}) {
  return _buildMedicalTopBar(
    toolbar: Stack(
      alignment: Alignment.center,
      children: [
        IconButton(
          onPressed: onClose,
          icon: const Icon(Icons.close),
          tooltip: '닫기',
          color: MedicalTokens.textMain,
        ),
        Align(
          alignment: Alignment.centerRight,
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              IconButton(
                onPressed: onToggleDraw,
                icon: Icon(drawMode ? Icons.edit : Icons.edit_outlined),
                tooltip: drawMode ? '그리기 끄기' : '그리기',
                color: drawMode ? MedicalTokens.primary : MedicalTokens.textMain,
                style: IconButton.styleFrom(
                  backgroundColor:
                      drawMode ? MedicalTokens.primarySoft : Colors.transparent,
                ),
              ),
              IconButton(
                onPressed: onClearAll,
                icon: const Icon(Icons.delete_outline),
                tooltip: '전체 지우기',
                color: MedicalTokens.textMain,
              ),
            ],
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
      backgroundColor: MedicalTokens.background,
      appBar: _buildMedicalTopBar(
        toolbar: Builder(
          builder: (toolbarContext) {
            final canPop = Navigator.of(toolbarContext).canPop();
            return Stack(
              alignment: Alignment.center,
              children: [
                if (canPop)
                  Align(
                    alignment: Alignment.centerLeft,
                    child: IconButton(
                      icon: Icon(
                        Theme.of(toolbarContext).platform ==
                                TargetPlatform.iOS
                            ? Icons.arrow_back_ios
                            : Icons.arrow_back,
                      ),
                      tooltip: MaterialLocalizations.of(toolbarContext)
                          .backButtonTooltip,
                      onPressed: () => Navigator.maybePop(toolbarContext),
                      color: MedicalTokens.textMain,
                    ),
                  ),
                IconButton(
                  onPressed: () {
                    _showZoomViewer(
                      toolbarContext,
                      storedBytes: original,
                      response: res,
                      explanationAbsoluteUrl: explanationAbsoluteUrl,
                    );
                  },
                  icon: const _MagnifierGlyph(size: 22),
                  tooltip: '이미지 확대 보기',
                  color: MedicalTokens.textMain,
                ),
                Align(
                  alignment: Alignment.centerRight,
                  child: Padding(
                    padding: const EdgeInsets.only(right: 4),
                    child: TextButton(
                      onPressed: () => _exportResultPdf(
                        toolbarContext,
                        original: original,
                        response: res,
                        explanationAbsoluteUrl: explanationAbsoluteUrl,
                      ),
                      child: const Text('내보내기'),
                    ),
                  ),
                ),
              ],
            );
          },
        ),
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
      return _FullscreenZoomViewer(
        storedBytes: storedBytes,
        response: response,
        explanationAbsoluteUrl: explanationAbsoluteUrl,
      );
    },
  );
}

/// 전체 화면 이미지 확대 — 닫기·펜(그리기) 토글.
class _FullscreenZoomViewer extends StatefulWidget {
  const _FullscreenZoomViewer({
    required this.storedBytes,
    required this.response,
    required this.explanationAbsoluteUrl,
  });

  final Uint8List? storedBytes;
  final AnalyzeResponse? response;
  final String? explanationAbsoluteUrl;

  @override
  State<_FullscreenZoomViewer> createState() => _FullscreenZoomViewerState();
}

class _FullscreenZoomViewerState extends State<_FullscreenZoomViewer> {
  bool _drawMode = false;
  final GlobalKey<_ZoomPanelState> _originalZoomKey = GlobalKey();
  final GlobalKey<_ZoomPanelState> _explanationZoomKey = GlobalKey();

  void _clearAllDrawings() {
    _originalZoomKey.currentState?.clearStrokes();
    _explanationZoomKey.currentState?.clearStrokes();
  }

  @override
  Widget build(BuildContext context) {
    return Dialog.fullscreen(
      child: Scaffold(
        backgroundColor: MedicalTokens.background,
        appBar: _buildZoomViewerTopBar(
          onClose: () => Navigator.pop(context),
          drawMode: _drawMode,
          onToggleDraw: () => setState(() => _drawMode = !_drawMode),
          onClearAll: _clearAllDrawings,
        ),
        body: Padding(
          padding: const EdgeInsets.all(MedicalTokens.spaceMd),
          child: LayoutBuilder(
            builder: (context, constraints) {
              final isWide = constraints.maxWidth >= 920;
              final originalPanel = _ZoomPanel(
                key: _originalZoomKey,
                title: '원본이미지',
                drawMode: _drawMode,
                child: _buildOriginalImageFit(
                  storedBytes: widget.storedBytes,
                  response: widget.response,
                ),
              );
              final url = widget.explanationAbsoluteUrl;
              final explanationPanel = _ZoomPanel(
                key: _explanationZoomKey,
                title: '결과 이미지',
                drawMode: _drawMode,
                child: (url != null && url.isNotEmpty)
                    ? Image.network(
                        url,
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
  }
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

class _ZoomPanel extends StatefulWidget {
  const _ZoomPanel({
    super.key,
    required this.title,
    required this.child,
    required this.drawMode,
  });

  final String title;
  final Widget child;
  final bool drawMode;

  @override
  State<_ZoomPanel> createState() => _ZoomPanelState();
}

class _ZoomPanelState extends State<_ZoomPanel> {
  final List<List<Offset>> _strokes = [];
  List<Offset>? _activeStroke;

  void _startStroke(Offset position) {
    setState(() => _activeStroke = [position]);
  }

  void _extendStroke(Offset position) {
    final stroke = _activeStroke;
    if (stroke == null) return;
    setState(() => _activeStroke = [...stroke, position]);
  }

  void _endStroke() {
    final stroke = _activeStroke;
    if (stroke == null || stroke.isEmpty) return;
    setState(() {
      _strokes.add(stroke);
      _activeStroke = null;
    });
  }

  void clearStrokes() {
    setState(() {
      _strokes.clear();
      _activeStroke = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final strokes = [
      ..._strokes,
      if (_activeStroke != null) _activeStroke!,
    ];

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(
          widget.title,
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
              child: Stack(
                fit: StackFit.expand,
                children: [
                  InteractiveViewer(
                    minScale: 1,
                    maxScale: 5,
                    panEnabled: !widget.drawMode,
                    scaleEnabled: !widget.drawMode,
                    child: ColoredBox(
                      color: Colors.black,
                      child: Center(child: widget.child),
                    ),
                  ),
                  if (widget.drawMode)
                    Listener(
                      behavior: HitTestBehavior.opaque,
                      onPointerDown: (event) =>
                          _startStroke(event.localPosition),
                      onPointerMove: (event) =>
                          _extendStroke(event.localPosition),
                      onPointerUp: (_) => _endStroke(),
                      onPointerCancel: (_) => _endStroke(),
                      child: CustomPaint(
                        painter: _WhiteStrokePainter(strokes: strokes),
                        child: const SizedBox.expand(),
                      ),
                    )
                  else if (strokes.isNotEmpty)
                    IgnorePointer(
                      child: CustomPaint(
                        painter: _WhiteStrokePainter(strokes: strokes),
                        child: const SizedBox.expand(),
                      ),
                    ),
                ],
              ),
            ),
          ),
        ),
      ],
    );
  }
}

class _WhiteStrokePainter extends CustomPainter {
  const _WhiteStrokePainter({required this.strokes});

  final List<List<Offset>> strokes;

  static const double _strokeWidth = 3;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = Colors.white
      ..strokeWidth = _strokeWidth
      ..strokeCap = StrokeCap.round
      ..strokeJoin = StrokeJoin.round
      ..style = PaintingStyle.stroke;

    for (final stroke in strokes) {
      if (stroke.isEmpty) continue;
      if (stroke.length == 1) {
        canvas.drawCircle(
          stroke.first,
          _strokeWidth / 2,
          Paint()
            ..color = Colors.white
            ..style = PaintingStyle.fill,
        );
        continue;
      }
      final path = Path()..moveTo(stroke.first.dx, stroke.first.dy);
      for (var i = 1; i < stroke.length; i++) {
        path.lineTo(stroke[i].dx, stroke[i].dy);
      }
      canvas.drawPath(path, paint);
    }
  }

  @override
  bool shouldRepaint(covariant _WhiteStrokePainter oldDelegate) =>
      oldDelegate.strokes != strokes;
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

/// 성능 지표 — 접었다 펼치기. `/analyze`·이력 `metrics`(eval_metrics)가 있을 때만 수치 표시.
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

    final metrics = res.modelPerformanceMetrics;
    final hasMetrics = metrics != null;
    final theme = Theme.of(context);
    final trailingLabel = _expanded
        ? '접기'
        : hasMetrics
            ? '펼쳐서 보기'
            : '지표 없음';

    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        InkWell(
          onTap: () => setState(() => _expanded = !_expanded),
          borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
          child: Padding(
            padding: const EdgeInsets.symmetric(vertical: 4),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Expanded(
                  child: Wrap(
                    crossAxisAlignment: WrapCrossAlignment.center,
                    spacing: 8,
                    runSpacing: 4,
                    children: [
                      Text(
                        '성능 지표',
                        style: theme.textTheme.titleMedium?.copyWith(
                          fontWeight: FontWeight.w700,
                          color: MedicalTokens.textMain,
                        ),
                      ),
                      if (hasMetrics)
                        Text(
                          '외부 테스트 배포 모델 평가 지표',
                          style: theme.textTheme.bodySmall?.copyWith(
                            color: MedicalTokens.textSubtle,
                            height: 1.45,
                          ),
                        ),
                    ],
                  ),
                ),
                Text(
                  trailingLabel,
                  style: theme.textTheme.bodySmall?.copyWith(
                    color: MedicalTokens.textSubtle,
                  ),
                ),
                const SizedBox(width: 4),
                Icon(
                  _expanded ? Icons.expand_less : Icons.expand_more,
                  size: 20,
                  color: MedicalTokens.textSubtle,
                ),
              ],
            ),
          ),
        ),
        if (_expanded) ...[
          const SizedBox(height: MedicalTokens.spaceSm),
          if (hasMetrics)
            _ReportMetricsCard(metrics: metrics)
          else
            const MedicalNoticeBanner(
              title: '성능 지표',
              body: 'AI eval_metrics가 응답에 없습니다.\n'
                  '서버 재시작 후에도 비어 있으면 '
                  'ai/artifacts/evaluations/ 평가 JSON을 확인하세요.',
            ),
        ],
        const SizedBox(height: MedicalTokens.spaceLg),
      ],
    );
  }
}

/// 백엔드 `/analyze` · 이력 `metrics` (AI external test eval_metrics).
class _ReportMetricsCard extends StatelessWidget {
  const _ReportMetricsCard({required this.metrics});

  final ReportMetrics metrics;

  @override
  Widget build(BuildContext context) {
    final rows = metrics.displayRows;
    final thresholds = metrics.thresholdDisplayRows;
    final xaiMeta = metrics.xaiMetaDisplayRows;
    final xaiRows = metrics.xaiDisplayRows;

    return _InfoCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _MetricSectionHeader(title: '분류 성능'),
          const SizedBox(height: 8),
          _MetricIndentedGroup(
            children: [
              for (var i = 0; i < rows.length; i++)
                _MetricRatioTile(
                  title: rows[i].title,
                  subtitle: rows[i].subtitle,
                  valueText: ReportMetrics.formatPercent(rows[i].ratio),
                  topGap: i > 0,
                ),
            ],
          ),
          const SizedBox(height: 16),
          _MetricSectionHeader(title: '\n임계값'),
          const SizedBox(height: 8),
          _MetricIndentedGroup(
            children: [
              for (var i = 0; i < thresholds.length; i++)
                _MetricThresholdTile(
                  title: thresholds[i].title,
                  subtitle: thresholds[i].subtitle,
                  valueText:
                      ReportMetrics.formatThreshold(thresholds[i].value),
                  topGap: i > 0,
                ),
            ],
          ),
          const SizedBox(height: 16),
          _MetricSectionHeader(title: '\nXAI 평가 설정'),
          const SizedBox(height: 8),
          _MetricIndentedGroup(
            children: [
              for (var i = 0; i < xaiMeta.length; i++)
                _MetricTextTile(
                  title: xaiMeta[i].title,
                  subtitle: xaiMeta[i].subtitle,
                  valueText: xaiMeta[i].value,
                  topGap: i > 0,
                ),
            ],
          ),
          const SizedBox(height: 16),
          _MetricSectionHeader(title: '\nXAI 지표'),
          const SizedBox(height: 8),
          _MetricIndentedGroup(
            children: [
              for (var i = 0; i < xaiRows.length; i++)
                _MetricRatioTile(
                  title: xaiRows[i].title,
                  subtitle: xaiRows[i].subtitle,
                  valueText: ReportMetrics.formatPercent(xaiRows[i].ratio),
                  topGap: i > 0,
                ),
            ],
          ),
        ],
      ),
    );
  }
}

class _MetricSectionHeader extends StatelessWidget {
  const _MetricSectionHeader({required this.title});

  final String title;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final base = theme.textTheme.titleSmall;
    return Text(
      title,
      style: base?.copyWith(
        fontSize: (base.fontSize ?? 14) + 1,
        fontWeight: FontWeight.w800,
        color: MedicalTokens.textMain,
        height: 1.3,
      ),
    );
  }
}

class _MetricIndentedGroup extends StatelessWidget {
  const _MetricIndentedGroup({required this.children});

  final List<Widget> children;

  static const double _indent = 18;

  @override
  Widget build(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.only(left: _indent),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: children,
      ),
    );
  }
}

class _MetricTextTile extends StatelessWidget {
  const _MetricTextTile({
    required this.title,
    required this.subtitle,
    required this.valueText,
    this.topGap = false,
  });

  final String title;
  final String subtitle;
  final String valueText;
  final bool topGap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (topGap) const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: MedicalTokens.textMain,
                ),
              ),
            ),
            MedicalBadge(
              text: valueText,
              backgroundColor: const Color(0xFFF0F4F8),
              foregroundColor: MedicalTokens.textMain,
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class _MetricRatioTile extends StatelessWidget {
  const _MetricRatioTile({
    required this.title,
    required this.subtitle,
    required this.valueText,
    this.topGap = false,
  });

  final String title;
  final String subtitle;
  final String valueText;
  final bool topGap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (topGap) const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: MedicalTokens.textMain,
                ),
              ),
            ),
            MedicalBadge(text: valueText),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
    );
  }
}

class _MetricThresholdTile extends StatelessWidget {
  const _MetricThresholdTile({
    required this.title,
    required this.subtitle,
    required this.valueText,
    this.topGap = false,
  });

  final String title;
  final String subtitle;
  final String valueText;
  final bool topGap;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        if (topGap) const SizedBox(height: 12),
        Row(
          children: [
            Expanded(
              child: Text(
                title,
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w700,
                  color: MedicalTokens.textMain,
                ),
              ),
            ),
            MedicalBadge(
              text: valueText,
              backgroundColor: const Color(0xFFF0F4F8),
              foregroundColor: MedicalTokens.textMain,
            ),
          ],
        ),
        const SizedBox(height: 4),
        Text(
          subtitle,
          style: theme.textTheme.bodySmall?.copyWith(
            color: theme.colorScheme.onSurfaceVariant,
          ),
        ),
      ],
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

