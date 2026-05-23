import 'package:flutter/material.dart';

import '../models/report_metrics.dart';
import 'dialog_keyboard.dart';
import 'medical_ui.dart';

/// 성능 지표 안내 다이얼로그 — Enter·Esc·바깥 탭·[X]로 닫기.
Future<void> showReportMetricsInfoDialog(
  BuildContext context, {
  ReportMetrics? metrics,
}) {
  final width = MediaQuery.sizeOf(context).width;
  final dialogWidth = (width - 48).clamp(280.0, 560.0);

  return showDialog<void>(
    context: context,
    barrierDismissible: true,
    builder: (dialogContext) {
      void close() => Navigator.of(dialogContext).pop();
      final theme = Theme.of(dialogContext);
      return dialogOkShortcuts(
        onClose: close,
        child: Dialog(
          child: ConstrainedBox(
          constraints: BoxConstraints(maxWidth: dialogWidth),
          child: Padding(
            padding: const EdgeInsets.fromLTRB(20, 16, 8, 20),
            child: Column(
              mainAxisSize: MainAxisSize.min,
              crossAxisAlignment: CrossAxisAlignment.stretch,
              children: [
                Row(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Expanded(
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            '성능 지표',
                            style: theme.textTheme.titleLarge?.copyWith(
                              fontWeight: FontWeight.w700,
                              color: MedicalTokens.textMain,
                            ),
                          ),
                          const SizedBox(height: 4),
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
                    IconButton(
                      onPressed: close,
                      icon: const Icon(Icons.close),
                    ),
                  ],
                ),
                const SizedBox(height: 8),
                ConstrainedBox(
                  constraints: BoxConstraints(
                    maxHeight: MediaQuery.sizeOf(dialogContext).height * 0.7,
                  ),
                  child: SingleChildScrollView(
                    child: ReportMetricsDialogBody(metrics: metrics),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
      );
    },
  );
}

/// 결과·업로드 화면 공통 — eval_metrics 표시 본문.
class ReportMetricsDialogBody extends StatelessWidget {
  const ReportMetricsDialogBody({super.key, this.metrics});

  final ReportMetrics? metrics;

  @override
  Widget build(BuildContext context) {
    final m = metrics;
    if (m == null || !m.hasDisplayableContent) {
      return const MedicalNoticeBanner(
        title: '성능 지표',
        body: 'AI eval_metrics가 응답에 없습니다.\n'
            '서버 재시작 후에도 비어 있으면 '
            'ai/artifacts/evaluations/ 평가 JSON을 확인하세요.',
      );
    }

    final rows = m.displayRows;
    final thresholds = m.thresholdDisplayRows;
    final xaiMeta = m.xaiMetaDisplayRows;
    final xaiRows = m.xaiDisplayRows;

    return MedicalCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const _MetricSectionHeader(title: '분류 성능'),
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
          const _MetricSectionHeader(title: '\n임계값'),
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
          const _MetricSectionHeader(title: '\nXAI 평가 설정'),
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
          const _MetricSectionHeader(title: '\nXAI 지표'),
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
