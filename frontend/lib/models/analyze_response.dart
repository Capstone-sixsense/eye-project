import 'report_metrics.dart';

/// 품질 정보(백엔드 `quality` 블록 등 — 필드는 API에 맞게 확장).
class QualitySummary {
  const QualitySummary({
    this.isAcceptable,
    this.warning,
    this.grade,
    this.gradeConfidence,
  });

  final bool? isAcceptable;
  final String? warning;
  final String? grade;
  final double? gradeConfidence;

  static QualitySummary? tryParse(dynamic json) {
    if (json is! Map<String, dynamic>) return null;
    final m = json;
    final low = m['is_low_quality'] as bool?;
    final acceptable = m['is_acceptable'] as bool?;
    return QualitySummary(
      isAcceptable: acceptable ?? (low != null ? !low : null),
      warning: m['warning'] as String? ?? m['quality_warning'] as String?,
      grade: m['grade'] as String? ?? m['quality_grade'] as String?,
      gradeConfidence: (m['grade_confidence'] as num?)?.toDouble() ??
          (m['quality_grade_confidence'] as num?)?.toDouble(),
    );
  }
}

/// `POST /analyze` 응답 JSON (스프린트 2 확장 필드는 선택적 — 없으면 기존 동작 유지).
class AnalyzeResponse {
  const AnalyzeResponse({
    required this.status,
    this.recordId,
    this.message,
    this.details,
    this.label,
    this.abnormalProbability,
    this.reportUrl,
    this.originalUrl,
    this.preprocessed,
    this.errorCode,
    this.quality,
    this.explanationImageUrl,
    this.xaiErrorCode,
    this.evalMetrics,
    this.decisionThreshold,
  });

  final String status;

  /// `POST /analyze` 의 `id` 또는 히스토리 메타의 `id` (예: `20260428_165403_123`).
  final String? recordId;
  final String? message;
  final Map<String, dynamic>? details;
  final String? label;
  final double? abnormalProbability;
  final String? reportUrl;
  final String? originalUrl;

  /// 전처리 통과 시 `1` 등. 없으면(구 API) 통과로 간주.
  final int? preprocessed;

  /// 본문 또는 `details`의 비즈니스 오류 코드 (`INPUT_CH_001` 등).
  final String? errorCode;

  final QualitySummary? quality;

  /// 설명 이미지(XAI/Grad-CAM 등) URL. `report_url`과 별도일 수 있음.
  final String? explanationImageUrl;

  /// 설명 이미지 생성 실패 시 `XAI_001` 등.
  final String? xaiErrorCode;

  /// external test 등 모델 평가 지표 (`metrics` / `eval_metrics`). 없으면 null.
  final ReportMetrics? evalMetrics;

  /// 이번 추론 판정 임계값 (`decision_threshold`).
  final double? decisionThreshold;

  /// 네트워크 이미지용 상대 경로. XAI 실패 시에는 null (실패 UI).
  String? get resolvedExplanationPath {
    if (shouldShowExplanationFailure) return null;
    if (explanationImageUrl != null && explanationImageUrl!.isNotEmpty) {
      return explanationImageUrl;
    }
    if (xaiErrorCode == null || xaiErrorCode!.isEmpty) {
      final r = reportUrl;
      if (r != null && r.isNotEmpty) return r;
    }
    return null;
  }

  bool get isSuccess => status == 'success';
  bool get isFail => status == 'fail';

  /// 판정·이상 확률(및 품질 블록) 표시 여부. 전처리 미통과 시 false.
  bool get canShowInferenceResults {
    if (isFail) return false;
    if (!isSuccess) return false;
    if (preprocessed != null && preprocessed != 1) return false;
    return true;
  }

  /// 설명 이미지 영역에 실패 메시지·코드를 보여야 하는지.
  bool get shouldShowExplanationFailure =>
      isSuccess &&
      canShowInferenceResults &&
      xaiErrorCode != null &&
      (explanationImageUrl == null || explanationImageUrl!.isEmpty);

  factory AnalyzeResponse.fromJson(Map<String, dynamic> json) {
    final details = json['details'] is Map<String, dynamic>
        ? json['details'] as Map<String, dynamic>
        : null;

    final explanation = json['explanation_url'] as String? ??
        json['explanation_image_url'] as String? ??
        json['heatmap_url'] as String?;

    final qualityRaw = json['quality'] ?? details?['quality'];

    return AnalyzeResponse(
      status: json['status'] as String? ?? 'unknown',
      recordId: json['id'] as String?,
      message: json['message'] as String?,
      details: details,
      label: json['label'] as String? ?? json['predicted_label'] as String?,
      abnormalProbability: (json['abnormal_probability'] as num?)?.toDouble(),
      reportUrl: json['report_url'] as String?,
      originalUrl: json['original_url'] as String?,
      preprocessed: _parsePreprocessed(json['preprocessed']),
      errorCode: json['error_code'] as String? ?? details?['error_code'] as String?,
      quality: QualitySummary.tryParse(qualityRaw),
      explanationImageUrl: explanation,
      xaiErrorCode:
          json['xai_error_code'] as String? ?? json['xai_error'] as String?,
      evalMetrics: _parseEvalMetrics(
        json,
        abnormalProbability: (json['abnormal_probability'] as num?)?.toDouble(),
      ),
      decisionThreshold: _readThreshold(json['decision_threshold']),
    );
  }

  /// `GET /history`·`GET /history/{id}` 에서 저장된 메타(JSON) 규격.
  factory AnalyzeResponse.fromHistoryRecord(Map<String, dynamic> json) {
    final abnormalProbability =
        (json['abnormal_probability'] as num?)?.toDouble();
    return AnalyzeResponse(
      status: 'success',
      recordId: json['id'] as String?,
      message: null,
      details: null,
      label: json['label'] as String?,
      abnormalProbability: abnormalProbability,
      reportUrl: (json['report_url'] ?? json['reportUrl']) as String?,
      originalUrl: (json['raw_url'] ?? json['original_url'] ?? json['rawUrl']) as String?,
      preprocessed: 1,
      errorCode: null,
      quality: QualitySummary.tryParse(json['quality']),
      explanationImageUrl: (json['explanation_url'] ??
          json['explanation_image_url'] ??
          json['heatmap_url']) as String?,
      xaiErrorCode: null,
      evalMetrics: ReportMetrics.tryParse(
        json['metrics'],
        abnormalProbability: abnormalProbability,
      ),
      decisionThreshold: _readThreshold(json['decision_threshold']) ??
          _readThreshold(
            (json['metrics'] is Map ? json['metrics'] as Map : null)?[
                'decision_threshold'],
          ),
    );
  }

  static ReportMetrics? _parseEvalMetrics(
    Map<String, dynamic> json, {
    double? abnormalProbability,
  }) {
    final fromMetrics = ReportMetrics.tryParse(
      json['metrics'],
      abnormalProbability: abnormalProbability,
    );
    if (fromMetrics != null) return fromMetrics;
    final fromEval = ReportMetrics.tryParse(
      json['eval_metrics'],
      abnormalProbability: abnormalProbability,
    );
    if (fromEval != null) return fromEval;
    final details = json['details'];
    if (details is Map<String, dynamic>) {
      return ReportMetrics.tryParse(
        details['eval_metrics'],
        abnormalProbability: abnormalProbability,
      );
    }
    return null;
  }

  static double? _readThreshold(dynamic value) {
    if (value is! num) return null;
    final n = value.toDouble();
    if (n.isNaN) return null;
    return n.clamp(0.0, 1.0);
  }

  static int? _parsePreprocessed(dynamic v) {
    if (v == null) return null;
    if (v is int) return v;
    if (v is bool) return v ? 1 : 0;
    if (v is num) return v.toInt();
    return null;
  }
}
