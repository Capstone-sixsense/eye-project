/// `POST /analyze` 응답 JSON.
class AnalyzeResponse {
  const AnalyzeResponse({
    required this.status,
    this.message,
    this.details,
    this.label,
    this.abnormalProbability,
    this.reportUrl,
    this.originalUrl,
  });

  final String status;
  final String? message;
  final Map<String, dynamic>? details;
  final String? label;
  final double? abnormalProbability;
  final String? reportUrl;
  final String? originalUrl;

  bool get isSuccess => status == 'success';
  bool get isFail => status == 'fail';

  factory AnalyzeResponse.fromJson(Map<String, dynamic> json) {
    return AnalyzeResponse(
      status: json['status'] as String? ?? 'unknown',
      message: json['message'] as String?,
      details: json['details'] is Map<String, dynamic>
          ? json['details'] as Map<String, dynamic>
          : null,
      label: json['label'] as String?,
      abnormalProbability: (json['abnormal_probability'] as num?)?.toDouble(),
      reportUrl: json['report_url'] as String?,
      originalUrl: json['original_url'] as String?,
    );
  }
}
