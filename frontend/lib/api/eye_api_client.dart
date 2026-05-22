import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;

import '../config/api_config.dart';
import '../models/analysis_history_entry.dart';
import '../models/analyze_response.dart';

/// FastAPI `detail`·JSON 본문에서 사용자용 메시지 추출.
String parseApiErrorMessage(String rawBody) {
  final trimmed = rawBody.trim();
  if (trimmed.isEmpty) return '알 수 없는 오류';
  try {
    final decoded = jsonDecode(trimmed);
    if (decoded is Map<String, dynamic>) {
      final fromDetail = _detailMessage(decoded['detail']);
      if (fromDetail != null && fromDetail.isNotEmpty) return fromDetail;
      final message = decoded['message'];
      if (message is String && message.isNotEmpty) return message;
    }
  } catch (_) {}
  if (trimmed.length <= 400 && !trimmed.startsWith('{')) return trimmed;
  return '요청을 처리할 수 없습니다.';
}

String? _detailMessage(dynamic detail) {
  if (detail is Map) {
    final message = detail['message'];
    if (message is String && message.isNotEmpty) return message;
  }
  if (detail is String && detail.isNotEmpty) return detail;
  if (detail is List && detail.isNotEmpty) {
    return detail.map((e) => e.toString()).join(', ');
  }
  return null;
}

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

class HistoryListPage {
  const HistoryListPage({
    required this.total,
    required this.limit,
    required this.offset,
    required this.items,
  });

  final int total;
  final int limit;
  final int offset;

  /// 이미 검증된 엔트리만 포함 (파싱 실패 행 제외).
  final List<AnalysisHistoryEntry> items;
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
    throw EyeApiException(response.statusCode, response.body, errorCode: code);
  }

  /// `GET /history` — 페이지네이션 목록 (최신순 서버 순서).
  Future<HistoryListPage> fetchHistoryPage({
    int limit = 50,
    int offset = 0,
  }) async {
    final base = ApiConfig.baseUrl.endsWith('/') ? ApiConfig.baseUrl : '${ApiConfig.baseUrl}/';
    final uri = Uri.parse(base).resolve('history').replace(
          queryParameters: <String, String>{
            'limit': '$limit',
            'offset': '$offset',
          },
        );

    debugPrint('[EyeApi] GET $uri');
    final response = await _client.get(uri);
    debugPrint('[EyeApi] history ${response.statusCode}');

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw EyeApiException(response.statusCode, response.body);
    }

    Map<String, dynamic>? map;
    try {
      final d = jsonDecode(response.body);
      if (d is Map<String, dynamic>) map = d;
    } catch (_) {}

    if (map == null) {
      throw EyeApiException(response.statusCode, '히스토리 응답 JSON 파싱 실패');
    }

    final rawItems = map['items'];
    final parsed = <AnalysisHistoryEntry>[];
    if (rawItems is List) {
      for (final e in rawItems) {
        if (e is Map) {
          final row = AnalysisHistoryEntry.tryParse(
            Map<String, dynamic>.from(e),
          );
          if (row != null) parsed.add(row);
        }
      }
    }

    return HistoryListPage(
      total: (map['total'] as num?)?.toInt() ?? parsed.length,
      limit: (map['limit'] as num?)?.toInt() ?? limit,
      offset: (map['offset'] as num?)?.toInt() ?? offset,
      items: parsed,
    );
  }

  Future<void> deleteHistoryRecord(String recordId) async {
    final base = ApiConfig.baseUrl.endsWith('/') ? ApiConfig.baseUrl : '${ApiConfig.baseUrl}/';
    final uri =
        Uri.parse(base).resolve('history/${Uri.encodeComponent(recordId)}');
    debugPrint('[EyeApi] DELETE $uri');
    final response = await _client.delete(uri);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw EyeApiException(response.statusCode, response.body);
    }
  }

  void close() => _client.close();
}
