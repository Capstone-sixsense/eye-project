import 'dart:typed_data';

import 'analyze_response.dart';

class AnalysisHistoryEntry {
  const AnalysisHistoryEntry({
    required this.recordId,
    required this.filename,
    required this.createdAt,
    required this.response,
    this.originalImageBytes,
  });

  final String recordId;
  final String filename;
  final DateTime createdAt;
  final AnalyzeResponse response;

  /// 방금 업로드 분석 후 세션에는 메모리 원본 유지 가능. 서버 목록에서는 null.
  final Uint8List? originalImageBytes;

  /// 서버 목록 행(`/history`) JSON → 로컬 엔트리.
  static AnalysisHistoryEntry? tryParse(Map<String, dynamic> json) {
    final id = (json['id'] ?? json['record_id']) as String?;
    if (id == null || id.isEmpty) return null;

    try {
      return AnalysisHistoryEntry(
        recordId: id,
        filename: (json['original_filename'] ?? json['filename']) as String? ?? 'image',
        createdAt: _parseCreated(
          (json['created_at'] ?? json['createdAt'] ?? json['timestamp']) as String?,
        ),
        response: AnalyzeResponse.fromHistoryRecord(json),
        originalImageBytes: null,
      );
    } catch (_) {
      return null;
    }
  }

  static DateTime _parseCreated(String? iso) {
    if (iso == null || iso.isEmpty) return DateTime.now();
    return DateTime.tryParse(iso)?.toLocal() ?? DateTime.now();
  }
}
