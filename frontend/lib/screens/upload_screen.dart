import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/eye_api_client.dart';
import '../constants/api_error_codes.dart';
import '../models/analyze_response.dart';
import '../models/report_metrics.dart';
import '../models/result_screen_args.dart';
import '../ui/analyze_progress_controller.dart';
import '../ui/analyze_progress_dialog.dart';
import '../ui/medical_ui.dart';
import '../ui/notice_dialog.dart'
    show showCodeNoticeDialog, showErrorNotice, showNoticeDialog;
import '../ui/report_metrics_dialog.dart';
import '../ui/server_logs_dialog.dart';

const int _kMaxUploadBytes = 10 * 1024 * 1024;
const Set<String> _kAllowedImageExtensions = {'jpg', 'jpeg', 'png'};

String? _normalizedExtension(PlatformFile f) {
  final fromPicker = f.extension?.trim().toLowerCase();
  if (fromPicker != null && fromPicker.isNotEmpty) {
    return fromPicker.startsWith('.') ? fromPicker.substring(1) : fromPicker;
  }
  final name = f.name;
  final dot = name.lastIndexOf('.');
  if (dot < 0 || dot >= name.length - 1) return null;
  return name.substring(dot + 1).toLowerCase();
}

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  String? fileName;
  Uint8List? fileBytes;
  bool _uploading = false;
  bool _deployMetricsLoading = true;
  ReportMetrics? _deployMetrics;

  final EyeApiClient _api = EyeApiClient();

  @override
  void initState() {
    super.initState();
    _loadDeployMetrics();
  }

  Future<void> _loadDeployMetrics() async {
    try {
      final metrics = await _api.fetchDeployMetrics();
      if (!mounted) return;
      setState(() {
        _deployMetrics = metrics;
        _deployMetricsLoading = false;
      });
    } on EyeApiException catch (e) {
      debugPrint('deploy-metric 실패: $e');
      if (!mounted) return;
      setState(() => _deployMetricsLoading = false);
    } catch (e, st) {
      debugPrint('deploy-metric 실패: $e\n$st');
      if (!mounted) return;
      setState(() => _deployMetricsLoading = false);
    }
  }

  Future<void> pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
      withData: true,
    );
    if (!mounted) return;
    if (result == null || result.files.isEmpty) return;

    final f = result.files.single;
    final ext = _normalizedExtension(f);
    if (ext == null || !_kAllowedImageExtensions.contains(ext)) {
      if (mounted) {
        await showNoticeDialog(
          context,
          message: 'jpg, jpeg, png 형식만 업로드할 수 있습니다.',
        );
      }
      return;
    }

    final int sizeBytes =
        f.size > 0 ? f.size : (f.bytes?.length ?? 0);
    if (sizeBytes > _kMaxUploadBytes) {
      if (mounted) {
        await showNoticeDialog(
          context,
          message: '10MB를 초과하는 파일은 업로드할 수 없습니다.',
        );
      }
      return;
    }

    final bytes = f.bytes;
    if (bytes == null) {
      if (mounted) {
        await showNoticeDialog(
          context,
          message:
              '이 브라우저에서 파일 데이터를 읽지 못했습니다. '
              '다른 이미지로 다시 시도하거나 크롬/엣지를 사용해 보세요.',
        );
      }
      return;
    }

    if (bytes.length > _kMaxUploadBytes) {
      if (mounted) {
        await showNoticeDialog(
          context,
          message: '10MB를 초과하는 파일은 업로드할 수 없습니다.',
        );
      }
      return;
    }

    setState(() {
      fileBytes = bytes;
      fileName = f.name;
    });
  }

  Future<void> _uploadAndAnalyze() async {
    final bytes = fileBytes;
    final name = fileName;
    if (bytes == null || name == null) return;

    setState(() => _uploading = true);
    final progressController = AnalyzeProgressController()..start();
    if (mounted) {
      showDialog<void>(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => PopScope(
          canPop: false,
          child: AlertDialog(
            content: ListenableBuilder(
              listenable: progressController,
              builder: (context, _) => AnalyzeProgressDialog(
                progress: progressController.visualProgress,
                phaseLabel: progressController.phaseLabel,
              ),
            ),
          ),
        ),
      );
    }
    try {
      final AnalyzeResponse res = await _api.analyze(
        bytes,
        name,
        onProgress: progressController.updateFromServer,
      );
      if (!mounted) return;

      if (res.errorCode == ApiErrorCodes.inputChannelUnsupported) {
        await showCodeNoticeDialog(
          context,
          code: ApiErrorCodes.inputChannelUnsupported,
          message: '4채널·CMYK 등은 분석할 수 없습니다.',
        );
        return;
      }

      await progressController.awaitVisualComplete();
      if (!mounted) return;

      await Navigator.pushNamed(
        context,
        '/result',
        arguments: ResultScreenArgs(
          analyzeResponse: res,
          originalImageBytes: bytes,
        ),
      );
    } on TimeoutException catch (e) {
      debugPrint('analyze 타임아웃: $e');
      if (!mounted) return;
      await showErrorNotice(context, e);
    } on EyeApiException catch (e) {
      if (!mounted) return;
      await showErrorNotice(context, e);
    } catch (e, st) {
      debugPrint('Upload/analyze 실패: $e\n$st');
      if (!mounted) return;
      await showErrorNotice(context, e);
    } finally {
      progressController.dispose();
      if (mounted) {
        final nav = Navigator.of(context, rootNavigator: true);
        if (nav.canPop()) nav.pop();
        setState(() => _uploading = false);
      }
    }
  }

  @override
  void dispose() {
    _api.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final width = MediaQuery.sizeOf(context).width;
    final contentMaxWidth = width > 680 ? 640.0 : 520.0;

    return Scaffold(
      appBar: AppBar(title: const Text('망막 이미지 분석')),
      body: SafeArea(
        child: Center(
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(MedicalTokens.spaceMd),
            child: ConstrainedBox(
              constraints: BoxConstraints(maxWidth: contentMaxWidth),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  Stack(
                    clipBehavior: Clip.none,
                    children: [
                      Padding(
                        padding: const EdgeInsets.only(right: 92),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              '이미지 업로드',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleMedium
                                  ?.copyWith(
                                    fontWeight: FontWeight.w700,
                                    color: MedicalTokens.textMain,
                                  ),
                            ),
                            const SizedBox(height: 4),
                            Text(
                              '선명한 안저 이미지를 선택한 뒤 분석을 진행하세요.',
                              style:
                                  Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color: MedicalTokens.textSubtle,
                                      ),
                            ),
                          ],
                        ),
                      ),
                      Positioned(
                        top: 0,
                        right: 0,
                        child: Row(
                          mainAxisSize: MainAxisSize.min,
                          children: [
                            _UploadHeaderIconButton(
                              tooltip: '서버 로그',
                              icon: Icons.list_rounded,
                              onPressed: () => showServerLogsDialog(context),
                            ),
                            const SizedBox(width: 4),
                            _UploadHeaderIconButton(
                              tooltip: '성능 지표',
                              onPressed: _deployMetricsLoading
                                  ? null
                                  : () => showReportMetricsInfoDialog(
                                        context,
                                        metrics: _deployMetrics,
                                      ),
                              icon: Icons.bar_chart_outlined,
                              loading: _deployMetricsLoading,
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: MedicalTokens.spaceMd),
                  const MedicalNoticeBanner(
                    title: '업로드 형식',
                    body: '업로드 가능한 확장자는 jpg, jpeg, png 3가지입니다.',
                  ),
                  const SizedBox(height: MedicalTokens.spaceMd),
                  MedicalCard(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        Row(
                          children: [
                            const Icon(
                              Icons.cloud_upload_outlined,
                              color: MedicalTokens.primary,
                            ),
                            const SizedBox(width: MedicalTokens.spaceXs),
                            Expanded(
                              child: Text(
                                fileName ?? '선택된 이미지가 없습니다',
                                style: Theme.of(context).textTheme.titleSmall,
                                overflow: TextOverflow.ellipsis,
                              ),
                            ),
                            if (fileBytes != null)
                              MedicalBadge(text: '${_kb(fileBytes!.length)} KB'),
                          ],
                        ),
                        const SizedBox(height: MedicalTokens.spaceMd),
                        _UploadPreview(fileBytes: fileBytes),
                        const SizedBox(height: MedicalTokens.spaceMd),
                        MedicalSecondaryButton(
                          label: '이미지 선택',
                          onPressed: _uploading ? null : pickFile,
                        ),
                        const SizedBox(height: MedicalTokens.spaceSm),
                        MedicalPrimaryButton(
                          label: _uploading ? '분석 진행 중...' : '업로드 및 분석',
                          onPressed: (fileBytes == null || _uploading)
                              ? null
                              : _uploadAndAnalyze,
                          leading: _uploading
                              ? const SizedBox(
                                  width: 18,
                                  height: 18,
                                  child: CircularProgressIndicator(
                                    strokeWidth: 2,
                                    color: Colors.white,
                                  ),
                                )
                              : const Icon(Icons.analytics_outlined, size: 18),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _UploadHeaderIconButton extends StatelessWidget {
  const _UploadHeaderIconButton({
    required this.tooltip,
    required this.icon,
    required this.onPressed,
    this.iconColor = MedicalTokens.textMain,
    this.loading = false,
  });

  final String tooltip;
  final IconData icon;
  final Color iconColor;
  final VoidCallback? onPressed;
  final bool loading;

  @override
  Widget build(BuildContext context) {
    return IconButton(
      tooltip: tooltip,
      onPressed: loading ? null : onPressed,
      style: IconButton.styleFrom(
        minimumSize: const Size(40, 40),
        fixedSize: const Size(40, 40),
        backgroundColor: MedicalTokens.surface,
        side: const BorderSide(color: MedicalTokens.border),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
        ),
      ),
      icon: loading
          ? const SizedBox(
              width: 20,
              height: 20,
              child: CircularProgressIndicator(strokeWidth: 2),
            )
          : Icon(
              icon,
              size: 22,
              color: onPressed == null ? MedicalTokens.textSubtle : iconColor,
            ),
    );
  }
}

class _UploadPreview extends StatelessWidget {
  const _UploadPreview({required this.fileBytes});

  final Uint8List? fileBytes;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        color: MedicalTokens.primarySoft.withValues(alpha: 0.45),
        borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
        border: Border.all(color: MedicalTokens.border),
      ),
      child: AspectRatio(
        aspectRatio: 1.25,
        child: fileBytes != null
            ? ClipRRect(
                borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
                child: Image.memory(fileBytes!, fit: BoxFit.contain),
              )
            : const Center(
                child: Text(
                  '이미지를 선택하면 미리보기가 표시됩니다.',
                  textAlign: TextAlign.center,
                ),
              ),
      ),
    );
  }
}

String _kb(int bytes) => (bytes / 1024).toStringAsFixed(1);

