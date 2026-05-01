import 'dart:typed_data';

import 'analyze_response.dart';

class ResultScreenArgs {
  const ResultScreenArgs({
    required this.analyzeResponse,
    this.originalImageBytes,
  });

  /// 히스토리 등에서 원본 바이트가 없으면 null — `analyzeResponse.originalUrl` 로 불러온다.
  final Uint8List? originalImageBytes;
  final AnalyzeResponse analyzeResponse;
}
