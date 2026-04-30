import 'dart:typed_data';

import 'analyze_response.dart';

class AnalysisHistoryEntry {
  const AnalysisHistoryEntry({
    required this.filename,
    required this.originalImageBytes,
    required this.response,
    required this.createdAt,
  });

  final String filename;
  final Uint8List originalImageBytes;
  final AnalyzeResponse response;
  final DateTime createdAt;
}
