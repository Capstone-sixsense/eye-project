import 'dart:async';
import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

import '../api/eye_api_client.dart';
import '../config/api_config.dart';
import '../models/analyze_response.dart';
import '../models/result_screen_args.dart';

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

      if (res.isFail) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text(res.message ?? '이미지 품질 미달')),
        );
        return;
      }

      if (!res.isSuccess) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(res.message ?? '분석 응답이 올바르지 않습니다. (status: ${res.status})'),
          ),
        );
        return;
      }

      await Navigator.pushNamed(
        context,
        '/result',
        arguments: ResultScreenArgs(
          originalImageBytes: bytes,
          analyzeResponse: res,
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
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('서버 오류 (${e.statusCode}): ${e.body}')),
      );
    } catch (e, st) {
      debugPrint('Upload/analyze 실패: $e\n$st');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text(
            '요청 실패: $e\n(API: ${ApiConfig.baseUrl} — 브라우저 주소창과 백엔드 포트를 확인하세요)',
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
    return Scaffold(
      appBar: AppBar(title: const Text('Upload Retinal Image')),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Text(
              'API: ${ApiConfig.baseUrl}',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Theme.of(context).colorScheme.outline,
                  ),
            ),
            const SizedBox(height: 8),
            if (fileBytes != null)
              Text(
                '선택된 파일: ${_kb(fileBytes!.length)} KB',
                style: Theme.of(context).textTheme.bodySmall,
              ),
            const SizedBox(height: 8),
            fileName != null
                ? Text('Selected: $fileName')
                : const Text('No image selected'),
            const SizedBox(height: 20),
            if (fileBytes != null)
              Image.memory(fileBytes!, width: 250, height: 250, fit: BoxFit.contain),

            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: _uploading ? null : pickFile,
              child: const Text('Select Image'),
            ),

            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: (fileBytes == null || _uploading) ? null : _uploadAndAnalyze,
              child: _uploading
                  ? const SizedBox(
                      width: 22,
                      height: 22,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Text('Upload'),
            ),
          ],
        ),
      ),
    );
  }
}

String _kb(int bytes) => (bytes / 1024).toStringAsFixed(1);
