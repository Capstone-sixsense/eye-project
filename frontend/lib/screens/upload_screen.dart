import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/eye_api_client.dart';
import '../constants/api_error_codes.dart';
import '../models/analyze_response.dart';
import '../models/result_screen_args.dart';
import '../ui/medical_ui.dart';

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

  final EyeApiClient _api = EyeApiClient();

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
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              'jpg, jpeg, png 형식만 업로드할 수 있습니다.',
            ),
            duration: Duration(seconds: 4),
          ),
        );
      }
      return;
    }

    final int sizeBytes =
        f.size > 0 ? f.size : (f.bytes?.length ?? 0);
    if (sizeBytes > _kMaxUploadBytes) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('10MB를 초과하는 파일은 업로드할 수 없습니다.'),
            duration: Duration(seconds: 4),
          ),
        );
      }
      return;
    }

    final bytes = f.bytes;
    if (bytes == null) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text(
              '이 브라우저에서 파일 데이터를 읽지 못했습니다. '
              '다른 이미지로 다시 시도하거나 크롬/엣지를 사용해 보세요.',
            ),
            duration: Duration(seconds: 5),
          ),
        );
      }
      return;
    }

    if (bytes.length > _kMaxUploadBytes) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
            content: Text('10MB를 초과하는 파일은 업로드할 수 없습니다.'),
            duration: Duration(seconds: 4),
          ),
        );
      }
      return;
    }

    setState(() {
      fileBytes = bytes;
      fileName = f.name;
    });
  }

  Future<void> _showInputChannelUnsupportedDialog() async {
    if (!mounted) return;
    await showDialog<void>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('지원하지 않는 이미지 형식'),
        content: SelectableText(
          '4채널·CMYK 등은 분석할 수 없습니다.\n\n${ApiErrorCodes.inputChannelUnsupported}',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx),
            child: const Text('확인'),
          ),
        ],
      ),
    );
  }

  Future<void> _uploadAndAnalyze() async {
    final bytes = fileBytes;
    final name = fileName;
    if (bytes == null || name == null) return;

    setState(() => _uploading = true);
    if (mounted) {
      showDialog<void>(
        context: context,
        barrierDismissible: false,
        builder: (ctx) => PopScope(
          canPop: false,
          child: AlertDialog(
            content: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                const CircularProgressIndicator(),
                const SizedBox(height: 20),
                Text(
                  '서버로 전송 후 AI 분석 중입니다.\n\n'
                  'Docker CPU 모드에서는 EfficientNet 추론에 '
                  '${EyeApiClient.analyzeTimeout.inMinutes}분 가까이 걸릴 수 있습니다. '
                  '백엔드 로그에 [analyze] 수신이 보이면 정상 처리 중입니다.',
                  style: Theme.of(ctx).textTheme.bodyMedium,
                ),
              ],
            ),
          ),
        ),
      );
    }
    try {
      final AnalyzeResponse res = await _api.analyze(bytes, name);
      if (!mounted) return;

      if (res.errorCode == ApiErrorCodes.inputChannelUnsupported) {
        await _showInputChannelUnsupportedDialog();
        return;
      }

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
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('$e'),
          duration: const Duration(seconds: 8),
        ),
      );
    } on EyeApiException catch (e) {
      if (!mounted) return;
      final isInputCh = e.errorCode == ApiErrorCodes.inputChannelUnsupported ||
          e.body.contains(ApiErrorCodes.inputChannelUnsupported);
      if (isInputCh) {
        await _showInputChannelUnsupportedDialog();
        return;
      }
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 오류 (${e.statusCode}): ${e.body}')),
      );
    } catch (e, st) {
      debugPrint('Upload/analyze 실패: $e\n$st');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '요청 실패: $e\n브라우저 주소창과 백엔드 포트를 확인하세요.',
          ),
          duration: const Duration(seconds: 6),
        ),
      );
    } finally {
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
                  const MedicalSectionTitle(
                    '이미지 업로드',
                    subtitle: '선명한 안저 이미지를 선택한 뒤 분석을 진행하세요.',
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

