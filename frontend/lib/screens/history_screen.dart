import 'package:flutter/material.dart';

/// 이력 보기 — 백엔드/스토리지 연동 전 플레이스홀더.
class HistoryScreen extends StatelessWidget {
  const HistoryScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('History')),
      body: const Center(
        child: Padding(
          padding: EdgeInsets.all(24),
          child: Text(
            '분석 이력은 추후 연동 예정입니다.',
            textAlign: TextAlign.center,
          ),
        ),
      ),
    );
  }
}
