/// 백엔드 `/analyze`·이력 `metrics` — AI `eval_metrics` (external test 요약).
class ReportMetrics {
  const ReportMetrics({
    this.accuracy,
    this.precision,
    this.sensitivity,
    this.specificity,
    this.f1,
    this.auroc,
    this.optimalThreshold,
    this.decisionThreshold,
    this.xaiEvalSplit,
    this.xaiEvalTargetBlock,
    this.xaiSegUnionIou,
    this.xaiSegMdice,
    this.xaiAucIou,
    this.xaiEvalN,
    this.isFromServer = false,
  });

  final bool isFromServer;

  final double? accuracy;
  final double? precision;
  final double? sensitivity;
  final double? specificity;
  final double? f1;
  final double? auroc;
  final double? optimalThreshold;
  final double? decisionThreshold;
  final String? xaiEvalSplit;
  final String? xaiEvalTargetBlock;
  final double? xaiSegUnionIou;
  final double? xaiSegMdice;
  final double? xaiAucIou;
  final int? xaiEvalN;

  static const List<({String key, String title, String subtitle})> _ratioRows = [
    (key: 'auroc', title: 'AUROC', subtitle: '판별력 — 임계값과 무관한 순위화 성능'),
    (key: 'accuracy', title: '정확도', subtitle: '최적 임계값에서 전체 맞춤 비율'),
    (key: 'sensitivity', title: '민감도 (재현율·TPR)', subtitle: '실제 이상을 이상으로 판정'),
    (key: 'specificity', title: '특이도 (TNR)', subtitle: '실제 정상을 정상으로 판정'),
    (key: 'precision', title: '정밀도 (PPV)', subtitle: '이상 판정 중 실제 이상 비율'),
    (key: 'f1', title: 'F1', subtitle: '정밀도와 민감도의 조화 평균'),
  ];

  static const List<({String key, String title, String subtitle})> _thresholdRows = [
    (
      key: 'optimal_threshold',
      title: '최적 임계값 (optimal_threshold)',
      subtitle: 'external test 평가에서 선택 (sensitivity+specificity−1 최대)',
    ),
    (
      key: 'decision_threshold',
      title: '판정 임계값 (decision_threshold)',
      subtitle: '추론 판정에 사용 — checkpoint → 평가 optimal → config → 0.5',
    ),
  ];

  static const List<({String key, String title, String subtitle})> _xaiMetaRows = [
    (key: 'xai_eval_split', title: 'XAI 평가 split', subtitle: 'xai_eval_split'),
    (
      key: 'xai_eval_target_block',
      title: 'XAI 평가 대상',
      subtitle: '병변 evidence 평가 블록 (xai_eval_target_block)',
    ),
    (key: 'xai_eval_n', title: 'XAI 평가 이미지 수', subtitle: 'xai_eval_n'),
  ];

  static const List<({String key, String title, String subtitle})> _xaiRatioRows = [
    (
      key: 'xai_seg_union_iou',
      title: '병변 후보 영역 IoU',
      subtitle: 'xai_seg_union_iou',
    ),
    (
      key: 'xai_seg_mdice',
      title: '병변 segmentation Dice',
      subtitle: 'xai_seg_mdice',
    ),
    (
      key: 'xai_auc_iou',
      title: 'AUC-IoU',
      subtitle: 'threshold sweep 안정성 지표',
    ),
  ];

  bool get hasDisplayableContent =>
      displayRows.isNotEmpty ||
      thresholdDisplayRows.isNotEmpty ||
      xaiMetaDisplayRows.isNotEmpty ||
      xaiDisplayRows.isNotEmpty;

  List<({String title, String subtitle, double ratio})> get displayRows {
    final out = <({String title, String subtitle, double ratio})>[];
    for (final row in _ratioRows) {
      final value = _ratioForKey(row.key);
      if (value != null) {
        out.add((title: row.title, subtitle: row.subtitle, ratio: value));
      }
    }
    return out;
  }

  List<({String title, String subtitle, double value})> get thresholdDisplayRows {
    final out = <({String title, String subtitle, double value})>[];
    for (final row in _thresholdRows) {
      final value = _thresholdForKey(row.key);
      if (value != null) {
        out.add((title: row.title, subtitle: row.subtitle, value: value));
      }
    }
    return out;
  }

  List<({String title, String subtitle, String value})> get xaiMetaDisplayRows {
    final out = <({String title, String subtitle, String value})>[];
    for (final row in _xaiMetaRows) {
      final value = _textForKey(row.key);
      if (value != null && value.isNotEmpty) {
        out.add((title: row.title, subtitle: row.subtitle, value: value));
      }
    }
    return out;
  }

  List<({String title, String subtitle, double ratio})> get xaiDisplayRows {
    final out = <({String title, String subtitle, double ratio})>[];
    for (final row in _xaiRatioRows) {
      final value = _ratioForKey(row.key);
      if (value != null) {
        out.add((title: row.title, subtitle: row.subtitle, ratio: value));
      }
    }
    return out;
  }

  double? _thresholdForKey(String key) => switch (key) {
        'optimal_threshold' => optimalThreshold,
        'decision_threshold' => decisionThreshold,
        _ => null,
      };

  String? _textForKey(String key) => switch (key) {
        'xai_eval_split' => xaiEvalSplit,
        'xai_eval_target_block' => xaiEvalTargetBlock,
        'xai_eval_n' => xaiEvalN?.toString(),
        _ => null,
      };

  double? _ratioForKey(String key) => switch (key) {
        'auroc' => auroc,
        'accuracy' => accuracy,
        'precision' => precision,
        'sensitivity' => sensitivity,
        'specificity' => specificity,
        'f1' => f1,
        'xai_seg_union_iou' => xaiSegUnionIou,
        'xai_seg_mdice' => xaiSegMdice,
        'xai_auc_iou' => xaiAucIou,
        _ => null,
      };

  static ReportMetrics? tryParse(
    dynamic json, {
    double? abnormalProbability,
  }) {
    if (json is! Map) return null;
    final m = Map<String, dynamic>.from(json);
    if (abnormalProbability != null &&
        _looksLikeProbabilityProxy(m, abnormalProbability)) {
      return null;
    }
    final parsed = _fromMap(m);
    return parsed.hasDisplayableContent
        ? parsed.copyWith(isFromServer: true)
        : null;
  }

  static ReportMetrics _fromMap(Map<String, dynamic> m) {
    return ReportMetrics(
      accuracy: _readRatio(m['accuracy']),
      precision: _readRatio(m['precision']),
      sensitivity: _readRatio(m['sensitivity']),
      specificity: _readRatio(m['specificity']),
      f1: _readRatio(m['f1']),
      auroc: _readRatio(m['auroc']),
      optimalThreshold: _readRatio(m['optimal_threshold']),
      decisionThreshold: _readRatio(m['decision_threshold']),
      xaiEvalSplit: m['xai_eval_split'] as String?,
      xaiEvalTargetBlock: m['xai_eval_target_block'] as String?,
      xaiSegUnionIou: _readRatio(m['xai_seg_union_iou']),
      xaiSegMdice: _readRatio(m['xai_seg_mdice']),
      xaiAucIou: _readRatio(m['xai_auc_iou']),
      xaiEvalN: _readInt(m['xai_eval_n']),
    );
  }

  ReportMetrics copyWith({
    double? decisionThreshold,
    bool? isFromServer,
  }) {
    return ReportMetrics(
      accuracy: accuracy,
      precision: precision,
      sensitivity: sensitivity,
      specificity: specificity,
      f1: f1,
      auroc: auroc,
      optimalThreshold: optimalThreshold,
      decisionThreshold: decisionThreshold ?? this.decisionThreshold,
      xaiEvalSplit: xaiEvalSplit,
      xaiEvalTargetBlock: xaiEvalTargetBlock,
      xaiSegUnionIou: xaiSegUnionIou,
      xaiSegMdice: xaiSegMdice,
      xaiAucIou: xaiAucIou,
      xaiEvalN: xaiEvalN,
      isFromServer: isFromServer ?? this.isFromServer,
    );
  }

  static bool _looksLikeProbabilityProxy(
    Map<String, dynamic> m,
    double prob,
  ) {
    final accuracy = _readRatio(m['accuracy']);
    final precision = _readRatio(m['precision']);
    final sensitivity = _readRatio(m['sensitivity']);
    final specificity = _readRatio(m['specificity']);
    final f1 = _readRatio(m['f1']);
    if (accuracy == null ||
        precision == null ||
        sensitivity == null ||
        specificity == null ||
        f1 == null) {
      return false;
    }
    const tolerance = 0.002;
    final matchesProb = (accuracy - prob).abs() < tolerance &&
        (precision - prob).abs() < tolerance &&
        (sensitivity - prob).abs() < tolerance &&
        (f1 - prob).abs() < tolerance;
    final matchesSpec = (specificity - (1.0 - prob)).abs() < tolerance;
    return matchesProb && matchesSpec;
  }

  static double? _readRatio(dynamic value) {
    if (value is! num) return null;
    final n = value.toDouble();
    if (n.isNaN) return null;
    if (n > 1.0) return (n / 100.0).clamp(0.0, 1.0);
    return n.clamp(0.0, 1.0);
  }

  static int? _readInt(dynamic value) {
    if (value is int) return value;
    if (value is num) return value.round();
    return null;
  }

  static String formatPercent(double ratio) =>
      '${(ratio * 100).toStringAsFixed(1)}%';

  static String formatThreshold(double value) => value.toStringAsFixed(2);
}
