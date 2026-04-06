import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../config/api_config.dart';
import '../models/analyze_response.dart';
import '../models/result_screen_args.dart';

class ResultScreen extends StatelessWidget {
  const ResultScreen({
    super.key,
    this.args,
    this.originalImageBytes,
  });

  /// `Upload` 이후 전달되는 인자.
  final ResultScreenArgs? args;

  /// 이전 라우트 호환(원본만).
  final Uint8List? originalImageBytes;

  @override
  Widget build(BuildContext context) {
    final Uint8List? original =
        args?.originalImageBytes ?? originalImageBytes;
    final AnalyzeResponse? res = args?.analyzeResponse;

    final String? reportUrl = res?.reportUrl;
    final String? reportAbsoluteUrl =
        (reportUrl != null && reportUrl.isNotEmpty)
            ? ApiConfig.resolveAssetUrl(reportUrl)
            : null;

    return Scaffold(
      appBar: AppBar(title: const Text('Result')),
      body: SafeArea(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              Expanded(
                child: Row(
                  crossAxisAlignment: CrossAxisAlignment.stretch,
                  children: [
                    Expanded(
                      child: _ImagePanel(
                        label: 'Original',
                        child: original != null
                            ? Image.memory(original, fit: BoxFit.contain)
                            : const Center(child: Text('No image')),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _ImagePanel(
                        label: 'Report (backend)',
                        child: reportAbsoluteUrl != null
                            ? Image.network(
                                reportAbsoluteUrl,
                                fit: BoxFit.contain,
                                loadingBuilder: (context, child, progress) {
                                  if (progress == null) return child;
                                  return const Center(
                                    child: CircularProgressIndicator(),
                                  );
                                },
                                errorBuilder: (context, error, stackTrace) => const Center(
                                  child: Text('이미지를 불러올 수 없습니다.\n(CORS 또는 URL 확인)'),
                                ),
                              )
                            : const Center(
                                child: Text('백엔드 report_url 없음'),
                              ),
                      ),
                    ),
                  ],
                ),
              ),
              const SizedBox(height: 16),
              const Text(
                'Medical metrics',
                style: TextStyle(fontWeight: FontWeight.bold),
              ),
              const SizedBox(height: 8),
              if (res != null && res.isSuccess) ...[
                Text('Label: ${res.label ?? "—"}'),
                Text(
                  'Abnormal probability: '
                  '${res.abnormalProbability != null ? "${(res.abnormalProbability! * 100).toStringAsFixed(1)}%" : "—"}',
                ),
              ] else
                const Text('—'),
            ],
          ),
        ),
      ),
    );
  }
}

class _ImagePanel extends StatelessWidget {
  const _ImagePanel({required this.label, required this.child});

  final String label;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        Text(label, style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 8),
        Expanded(
          child: DecoratedBox(
            decoration: BoxDecoration(
              border: Border.all(color: Theme.of(context).dividerColor),
              borderRadius: BorderRadius.circular(8),
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(8),
              child: child,
            ),
          ),
        ),
      ],
    );
  }
}
