import 'dart:typed_data';

import 'package:flutter/material.dart';

/// 업로드 원본은 좌측, 백엔드 Grad-CAM 등은 우측·하단에 연동 예정.
class ResultScreen extends StatelessWidget {
  const ResultScreen({super.key, this.originalImageBytes});

  final Uint8List? originalImageBytes;

  @override
  Widget build(BuildContext context) {
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
                        child: originalImageBytes != null
                            ? Image.memory(
                                originalImageBytes!,
                                fit: BoxFit.contain,
                              )
                            : const Center(child: Text('No image')),
                      ),
                    ),
                    const SizedBox(width: 16),
                    Expanded(
                      child: _ImagePanel(
                        label: 'Grad-CAM (from API)',
                        child: const Center(
                          child: Text('Waiting for backend response'),
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
              const Text('— API 연동 후 표시'),
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
