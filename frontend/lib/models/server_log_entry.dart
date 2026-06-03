/// `GET /logs` 응답 항목.
class ServerLogEntry {
  const ServerLogEntry({
    required this.id,
    required this.ts,
    required this.level,
    required this.message,
    this.phase,
    this.jobId,
    this.elapsed,
  });

  final int id;
  final String ts;
  final String level;
  final String? phase;
  final String? jobId;
  final String message;
  final double? elapsed;

  static ServerLogEntry? tryParse(Map<String, dynamic> json) {
    final id = json['id'];
    final ts = json['ts'];
    final level = json['level'];
    final message = json['message'];
    if (id is! num || ts is! String || level is! String || message is! String) {
      return null;
    }
    return ServerLogEntry(
      id: id.toInt(),
      ts: ts,
      level: level,
      phase: json['phase'] as String?,
      jobId: json['job_id'] as String?,
      message: message,
      elapsed: (json['elapsed'] as num?)?.toDouble(),
    );
  }

  /// 진행 단계·로그 phase용 한국어 문구.
  String? get phaseLabel {
    switch (phase) {
      case 'startup':
        return '서버 기동';
      case 'upload':
        return '업로드';
      case 'fundus_check':
        return '안저 검증';
      case 'quickqual':
        return '품질 평가';
      case 'inference':
        return 'AI 분석';
      case 'report':
        return '리포트';
      case 'done':
        return '완료';
      default:
        return phase;
    }
  }
}
