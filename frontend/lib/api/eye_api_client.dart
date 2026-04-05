import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/analyze_response.dart';

class EyeApiException implements Exception {
  EyeApiException(this.statusCode, this.body);

  final int statusCode;
  final String body;

  @override
  String toString() => 'EyeApiException($statusCode): $body';
}

/// `POST /analyze` — 백엔드 파라미터명 `image`와 일치.
class EyeApiClient {
  EyeApiClient({http.Client? httpClient}) : _client = httpClient ?? http.Client();

  final http.Client _client;

  /// CPU 추론·대용량 업로드까지 포함. 기본 20분.
  static const Duration analyzeTimeout = Duration(minutes: 20);

  Future<AnalyzeResponse> analyze(Uint8List imageBytes, String filename) async {
    return _analyzeWithTimeout(imageBytes, filename);
  }

  Future<AnalyzeResponse> _analyzeWithTimeout(
    Uint8List imageBytes,
    String filename,
  ) async {
    final uri = Uri.parse('${ApiConfig.baseUrl}/analyze');
    debugPrint(
      '[EyeApi] POST $uri  (파일: $filename, ${imageBytes.length} bytes, 필드명: image)',
    );
    final request = http.MultipartRequest('POST', uri);
    request.files.add(
      http.MultipartFile.fromBytes('image', imageBytes, filename: filename),
    );

    late http.StreamedResponse streamed;
    try {
      streamed = await _client.send(request).timeout(
        analyzeTimeout,
        onTimeout: () => throw TimeoutException(
          '서버가 ${analyzeTimeout.inMinutes}분 안에 응답하지 않았습니다. '
          'Docker CPU 모드는 추론에 매우 오래 걸릴 수 있습니다.',
        ),
      );
    } on TimeoutException catch (e) {
      debugPrint('[EyeApi] send 타임아웃: $e');
      rethrow;
    }

    late http.Response response;
    try {
      response = await http.Response.fromStream(streamed).timeout(
        const Duration(minutes: 2),
        onTimeout: () => throw TimeoutException('응답 본문 수신 시간 초과'),
      );
    } on TimeoutException catch (e) {
      debugPrint('[EyeApi] 본문 수신 타임아웃: $e');
      rethrow;
    }

    Map<String, dynamic>? jsonMap;
    try {
      final decoded = jsonDecode(response.body);
      if (decoded is Map<String, dynamic>) {
        jsonMap = decoded;
      }
    } catch (_) {}

    if (response.statusCode >= 200 && response.statusCode < 300 && jsonMap != null) {
      debugPrint('[EyeApi] 응답 ${response.statusCode}, keys: ${jsonMap.keys.toList()}');
      return AnalyzeResponse.fromJson(jsonMap);
    }

    debugPrint(
      '[EyeApi] 비정상 응답 ${response.statusCode}, body(앞 200자): '
      '${response.body.length > 200 ? "${response.body.substring(0, 200)}..." : response.body}',
    );
    final detail = jsonMap?['detail'];
    final msg = detail is String
        ? detail
        : detail is List
            ? detail.map((e) => e.toString()).join(', ')
            : response.body;
    throw EyeApiException(response.statusCode, msg);
  }

  void close() => _client.close();
}
