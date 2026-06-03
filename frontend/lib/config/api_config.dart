/// 백엔드 베이스 URL.
/// Windows 등에서 `localhost`가 IPv6(::1)만 타면 Docker 포트와 안 맞을 수 있어 기본은 127.0.0.1.
/// 예: `flutter run --dart-define=API_BASE_URL=http://192.168.0.10:8000`
abstract final class ApiConfig {
  static const String baseUrl = String.fromEnvironment(
    'API_BASE_URL',
    defaultValue: 'http://127.0.0.1:8000',
  );

  /// API 요청용 베이스 URI (`/` 로 끝남).
  static Uri get baseUri {
    final base = baseUrl.endsWith('/') ? baseUrl : '$baseUrl/';
    return Uri.parse(base);
  }

  /// 서버가 주는 경로(`results/...` 등)를 이미지 요청용 절대 URL로 변환.
  static String resolveAssetUrl(String serverPath) {
    final p = serverPath.replaceAll(r'\', '/').replaceFirst(RegExp(r'^/+'), '');
    return baseUri.resolve(p).toString();
  }
}
