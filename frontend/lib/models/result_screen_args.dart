import 'dart:typed_data';

import 'analyze_response.dart';

class ResultScreenArgs {
  const ResultScreenArgs({
    required this.originalImageBytes,
    required this.analyzeResponse,
  });

  final Uint8List originalImageBytes;
  final AnalyzeResponse analyzeResponse;
}
