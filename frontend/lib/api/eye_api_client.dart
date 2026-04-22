import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/analyze_response.dart';

/// 백엔드/게이트웨이가 돌려주는 HTTP 오류 본문에서 코드 추출 (가능할 때만).
String? parseErrorCodeFromBody(String body) {
  try {
    final decoded = jsonDecode(body);
    if (decoded is Map<String, dynamic>) {
      final direct = decoded['error_code'] as String?;
      if (direct != null && direct.isNotEmpty) return direct;
      final detail = decoded['detail'];
      if (detail is String && detail.isNotEmpty) {
        if (RegExp(r'^[A-Z0-9_]+$').hasMatch(detail.trim())) {
          return detail.trim();
        }
      }
      if (detail is List && detail.isNotEmpty) {
        final first = detail.first;
        if (first is Map && first['msg'] is String) {
          final msg = (first['msg'] as String).trim();
          final m = RegExp(r'\b([A-Z][A-Z0-9_]+)\b').firstMatch(msg);
          if (m != null) return m.group(1);
        }
      }
    }
  } catch (_) {}
  return null;
}

class EyeApiException implements Exception {
  EyeApiException(this.statusCode, this.body, {this.errorCode});

  final int statusCode;
  final String body;
  final String? errorCode;

  @override
  String toString() =>
      'EyeApiException($statusCode)${errorCode != null ? '[$errorCode]' : ''}: $body';
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
    final code = parseErrorCodeFromBody(response.body);
    final detail = jsonMap?['detail'];
    final msg = detail is String
        ? detail
        : detail is List
            ? detail.map((e) => e.toString()).join(', ')
            : response.body;
    throw EyeApiException(response.statusCode, msg, errorCode: code);
  }

  void close() => _client.close();
}
