import 'dart:typed_data';

import 'package:flutter/material.dart';

import '../config/api_config.dart';
import '../constants/api_error_codes.dart';
import '../models/analyze_response.dart';
import '../models/result_screen_args.dart';

/// 원본·설명 이미지 타일 높이를 맞추기 위한 공통 상한.
const double _kResultImageMaxHeight = 280;

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

    return Scaffold(
      appBar: AppBar(title: const Text('Analysis result')),
      body: SafeArea(
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(16),
                child: LayoutBuilder(
                  builder: (context, constraints) {
                    final wide = constraints.maxWidth >= 720;
                    final originalBox = _ImageBox(
                      child: original != null
                          ? Image.memory(original, fit: BoxFit.contain)
                          : const Center(child: Text('No image')),
                    );
                    final aiPanel = _ExplanationPanel(
                      absoluteUrl: explanationAbsoluteUrl,
                      response: res,
                    );
                    final judgment = _JudgmentCard(response: res);
                    final reportMetrics = _ReportMetricsCard(response: res);
                    final probQuality = _ScoreQualityCard(response: res);

                    if (wide) {
                      return Column(
                        crossAxisAlignment: CrossAxisAlignment.stretch,
                        children: [
                          Row(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.stretch,
                                  children: [
                                    const _SectionTitle('Original image'),
                                    const SizedBox(height: 8),
                                    originalBox,
                                  ],
                                ),
                              ),
                              const SizedBox(width: 20),
                              Expanded(
                                child: Column(
                                  crossAxisAlignment: CrossAxisAlignment.stretch,
                                  children: [
                                    const _SectionTitle('Explanation image'),
                                    const SizedBox(height: 8),
                                    aiPanel,
                                  ],
                                ),
                              ),
                            ],
                          ),
                          const SizedBox(height: 24),
                          const _SectionTitle('Judgment'),
                          const SizedBox(height: 8),
                          judgment,
                          const SizedBox(height: 20),
                          const _SectionTitle('Report metrics'),
                          const SizedBox(height: 8),
                          reportMetrics,
                          const SizedBox(height: 20),
                          const _SectionTitle('Anomaly probability & quality'),
                          const SizedBox(height: 8),
                          probQuality,
                        ],
                      );
                    }

                    return Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      children: [
                        const _SectionTitle('Original image'),
                        const SizedBox(height: 8),
                        originalBox,
                        const SizedBox(height: 20),
                        const _SectionTitle('Explanation image'),
                        const SizedBox(height: 8),
                        aiPanel,
                        const SizedBox(height: 20),
                        const _SectionTitle('Judgment'),
                        const SizedBox(height: 8),
                        judgment,
                        const SizedBox(height: 20),
                        const _SectionTitle('Report metrics'),
                        const SizedBox(height: 8),
                        reportMetrics,
                        const SizedBox(height: 20),
                        const _SectionTitle('Anomaly probability & quality'),
                        const SizedBox(height: 8),
                        probQuality,
                      ],
                    );
                  },
                ),
              ),
            ),
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 8, 16, 16),
              child: Row(
                children: [
                  Expanded(
                    child: OutlinedButton(
                      onPressed: () {
                        Navigator.pushNamedAndRemoveUntil(
                          context,
                          '/upload',
                          (route) => false,
                        );
                      },
                      child: const Text('다시 업로드'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: FilledButton(
                      onPressed: () {
                        Navigator.pushNamed(context, '/history');
                      },
                      child: const Text('이력 보기'),
                    ),
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _SectionTitle extends StatelessWidget {
  const _SectionTitle(this.text);

  final String text;

  @override
  Widget build(BuildContext context) {
    return Text(
      text,
      style: Theme.of(context).textTheme.titleMedium?.copyWith(
            fontWeight: FontWeight.w600,
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
      return const _InfoCard(child: Text('—'));
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
      child: Text(
        res.label ?? '—',
        style: Theme.of(context).textTheme.titleLarge,
      ),
    );
  }
}

/// 백엔드 `make_result_img` / `analyze` 의 metrics dict와 동일한 5지표.
class _ReportMetricsCard extends StatelessWidget {
  const _ReportMetricsCard({required this.response});

  final AnalyzeResponse? response;

  static const List<(String, String)> _labels = [
    ('Accuracy', '전체 성능 판단'),
    ('Precision', '불필요 오진 최소화'),
    ('Sensitivity', '놓치는 환자 최소화'),
    ('Specificity', '정상 오진 방지'),
    ('F1-score', '정밀도와 재현율 조화'),
  ];

  static double _metricValue(int index, double prob) {
    if (index == 3) {
      return 1.0 - prob;
    }
    return prob;
  }

  @override
  Widget build(BuildContext context) {
    final res = response;
    if (res == null) {
      return const _InfoCard(child: Text('—'));
    }
    if (res.isFail || !res.canShowInferenceResults) {
      return const _InfoCard(
        child: Text(
          '전처리를 통과하지 않아 리포트 지표를 표시할 수 없습니다.',
        ),
      );
    }

    final prob = res.abnormalProbability;
    if (prob == null) {
      return const _InfoCard(
        child: Text('Abnormal probability가 없어 리포트 지표를 계산할 수 없습니다.'),
      );
    }

    final theme = Theme.of(context);
    return _InfoCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          for (var i = 0; i < _labels.length; i++) ...[
            if (i > 0) const SizedBox(height: 14),
            Text(
              '${_labels[i].$1}: ${(_metricValue(i, prob) * 100).toStringAsFixed(1)}%',
              style: theme.textTheme.titleSmall?.copyWith(
                fontWeight: FontWeight.w600,
                color: const Color(0xFF003366),
              ),
            ),
            const SizedBox(height: 4),
            Text(
              '(${_labels[i].$2})',
              style: theme.textTheme.bodySmall?.copyWith(
                color: theme.colorScheme.onSurfaceVariant,
              ),
            ),
          ],
        ],
      ),
    );
  }
}

class _ScoreQualityCard extends StatelessWidget {
  const _ScoreQualityCard({required this.response});

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

    final q = res.quality;
    final prob = res.abnormalProbability;

    return _InfoCard(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            prob != null
                ? 'Abnormal probability: ${(prob * 100).toStringAsFixed(1)}%'
                : 'Abnormal probability: —',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          if (q != null) ...[
            const SizedBox(height: 12),
            if (q.grade != null)
              Text('Quality grade: ${q.grade}'
                  '${q.gradeConfidence != null ? " (${(q.gradeConfidence! * 100).toStringAsFixed(0)}% conf.)" : ""}'),
            if (q.isAcceptable != null)
              Text('Acceptable: ${q.isAcceptable! ? "yes" : "no"}'),
            if (q.warning != null && q.warning!.isNotEmpty)
              Text(
                q.warning!,
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.onSurfaceVariant,
                    ),
              ),
          ] else
            Padding(
              padding: const EdgeInsets.only(top: 8),
              child: Text(
                'No quality block in response (legacy API).',
                style: Theme.of(context).textTheme.bodySmall?.copyWith(
                      color: Theme.of(context).colorScheme.outline,
                    ),
              ),
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
              'Explanation image generation failed',
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
        child: DecoratedBox(
          decoration: BoxDecoration(
            border: Border.all(color: Theme.of(context).dividerColor),
            borderRadius: BorderRadius.circular(8),
          ),
          child: ClipRRect(
            borderRadius: BorderRadius.circular(8),
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
      );
    }

    return const _InfoCard(
      child: Text('No explanation image URL in response.'),
    );
  }
}

class _InfoCard extends StatelessWidget {
  const _InfoCard({required this.child});

  final Widget child;

  @override
  Widget build(BuildContext context) {
    return DecoratedBox(
      decoration: BoxDecoration(
        border: Border.all(color: Theme.of(context).dividerColor),
        borderRadius: BorderRadius.circular(8),
        color: Theme.of(context)
            .colorScheme
            .surfaceContainerHighest
            .withValues(alpha: 0.35),
      ),
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: child,
      ),
    );
  }
}
