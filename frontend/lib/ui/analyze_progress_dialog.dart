import 'package:flutter/material.dart';

import 'medical_ui.dart';

/// 분석 대기 UI — `progress`는 추후 백엔드 진행 신호 연동 시 0.0~1.0.
class AnalyzeProgressDialog extends StatelessWidget {
  const AnalyzeProgressDialog({super.key, this.progress});

  /// `null`이면 미정(indeterminate). 백엔드 연동 후 단계별 값 설정.
  final double? progress;

  static const String message = '서버로 전송 후 AI 분석중입니다.';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return SizedBox(
      width: 280,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: progress,
              minHeight: 4,
              backgroundColor: MedicalTokens.border,
              color: MedicalTokens.primary,
            ),
          ),
          const SizedBox(height: 20),
          Text(
            message,
            textAlign: TextAlign.center,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: MedicalTokens.textMain,
            ),
          ),
        ],
      ),
    );
  }
}
