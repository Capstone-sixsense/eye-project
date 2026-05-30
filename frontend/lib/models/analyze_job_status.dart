import 'analyze_response.dart';

/// `GET /analyze/jobs/{job_id}` 응답.
class AnalyzeJobStatus {
  const AnalyzeJobStatus({
    required this.status,
    required this.progress,
    this.phase,
    this.result,
    this.error,
  });

  /// `queued` | `running` | `done` | `failed`
  final String status;

  /// 0.0 ~ 1.0
  final double progress;

  /// `upload`, `fundus_check`, `quickqual`, `inference`, `report`, `done` 등
  final String? phase;

  final AnalyzeResponse? result;

  /// 실패 시 `{ "status_code": int, "detail": ... }`
  final Map<String, dynamic>? error;

  bool get isDone => status == 'done';
  bool get isFailed => status == 'failed';

  factory AnalyzeJobStatus.fromJson(Map<String, dynamic> json) {
    AnalyzeResponse? result;
    final rawResult = json['result'];
    if (rawResult is Map<String, dynamic>) {
      result = AnalyzeResponse.fromJson(rawResult);
    }

    Map<String, dynamic>? error;
    final rawError = json['error'];
    if (rawError is Map<String, dynamic>) {
      error = rawError;
    }

    return AnalyzeJobStatus(
      status: json['status'] as String? ?? 'unknown',
      progress: (json['progress'] as num?)?.toDouble().clamp(0.0, 1.0) ?? 0.0,
      phase: json['phase'] as String?,
      result: result,
      error: error,
    );
  }

  /// 진행 다이얼로그용 한국어 단계 문구.
  String get phaseLabel {
    switch (phase) {
      case 'upload':
        return '이미지 확인 중';
      case 'fundus_check':
        return '안저 이미지 검증 중';
      case 'quickqual':
        return '이미지 품질 평가 중';
      case 'inference':
        return 'AI 분석 중';
      case 'report':
        return '리포트 생성 중';
      case 'done':
        return '완료';
    }
    switch (status) {
      case 'queued':
        return '분석 대기 중';
      case 'running':
        return '분석 진행 중';
      case 'failed':
        return '분석 실패';
      default:
        return '서버로 전송 후 AI 분석 중';
    }
  }
}
