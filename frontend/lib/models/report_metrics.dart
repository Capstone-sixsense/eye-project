/// 백엔드 `/analyze`·이력 `metrics` — AI `eval_metrics` (external test 요약).
class ReportMetrics {
  const ReportMetrics({
    this.accuracy,
    this.precision,
    this.sensitivity,
    this.specificity,
    this.f1,
    this.auroc,
    this.decisionThreshold,
  });

  final double? accuracy;
  final double? precision;
  final double? sensitivity;
  final double? specificity;
  final double? f1;
  final double? auroc;
  final double? decisionThreshold;

  static const List<({String key, String title, String subtitle})> _rows = [
    (key: 'accuracy', title: 'Accuracy', subtitle: '전체 성능 판단'),
    (key: 'precision', title: 'Precision', subtitle: '불필요 오진 최소화'),
    (key: 'sensitivity', title: 'Sensitivity', subtitle: '놓치는 환자 최소화'),
    (key: 'specificity', title: 'Specificity', subtitle: '정상 오진 방지'),
    (key: 'f1', title: 'F1-score', subtitle: '정밀도와 재현율 조화'),
  ];

  bool get hasClassificationMetrics =>
      accuracy != null ||
      precision != null ||
      sensitivity != null ||
      specificity != null ||
      f1 != null;

  /// UI·PDF에 표시할 (제목, 설명, 0~1 비율) 목록. 값이 있는 항목만 포함.
  List<({String title, String subtitle, double ratio})> get displayRows {
    final out = <({String title, String subtitle, double ratio})>[];
    for (final row in _rows) {
      final value = _valueForKey(row.key);
      if (value != null) {
        out.add((title: row.title, subtitle: row.subtitle, ratio: value));
      }
    }
    return out;
  }

  double? _valueForKey(String key) => switch (key) {
        'accuracy' => accuracy,
        'precision' => precision,
        'sensitivity' => sensitivity,
        'specificity' => specificity,
        'f1' => f1,
        _ => null,
      };

  /// [abnormalProbability]가 있으면, 예전 백엔드가 확률로 채우던 가짜 metrics는 무시한다.
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
    final parsed = ReportMetrics(
      accuracy: _readRatio(m['accuracy']),
      precision: _readRatio(m['precision']),
      sensitivity: _readRatio(m['sensitivity']),
      specificity: _readRatio(m['specificity']),
      f1: _readRatio(m['f1']),
      auroc: _readRatio(m['auroc']),
      decisionThreshold: _readRatio(m['decision_threshold']),
    );
    return parsed.hasClassificationMetrics ? parsed : null;
  }

  /// 예전 placeholder: accuracy/precision/sensitivity/f1 ≈ prob, specificity ≈ 1−prob
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

  static String formatPercent(double ratio) =>
      '${(ratio * 100).toStringAsFixed(1)}%';
}
