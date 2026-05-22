import '../models/analyze_response.dart';
import '../models/report_metrics.dart';

/// 마지막 `/analyze` 응답의 배포 eval_metrics — 업로드 화면 성능 지표용 (프론트 전용).
class DeployMetricsStore {
  DeployMetricsStore._();

  static ReportMetrics? cached;

  static void updateFromAnalyze(AnalyzeResponse response) {
    cached = response.modelPerformanceMetrics;
  }
}
