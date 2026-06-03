import 'package:flutter/material.dart';

import 'medical_ui.dart';

/// 분석 대기 UI — 백엔드 `progress`·`phase`와 연동.
class AnalyzeProgressDialog extends StatelessWidget {
  const AnalyzeProgressDialog({
    super.key,
    this.progress = 0,
    this.phaseLabel,
  });

  /// 0.0~1.0 UI 표시용 (보간된 값).
  final double progress;

  /// `AnalyzeJobStatus.phaseLabel` 등 단계 설명.
  final String? phaseLabel;

  static const String defaultMessage = '서버로 전송 후 AI 분석 중입니다.';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final label = phaseLabel?.trim();
    final message = (label != null && label.isNotEmpty) ? label : defaultMessage;
    final value = progress.clamp(0.0, 1.0);
    final percent = (value * 100).round().clamp(0, 100);

    return SizedBox(
      width: 280,
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          ClipRRect(
            borderRadius: BorderRadius.circular(4),
            child: LinearProgressIndicator(
              value: value < 0.01 ? 0.01 : value,
              minHeight: 4,
              backgroundColor: MedicalTokens.border,
              color: MedicalTokens.primary,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            '$percent%',
            textAlign: TextAlign.center,
            style: theme.textTheme.labelSmall?.copyWith(
              color: MedicalTokens.textSubtle,
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
