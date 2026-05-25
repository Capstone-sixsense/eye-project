import 'dart:async';

import 'package:flutter/foundation.dart';

import '../models/analyze_job_status.dart';

/// 서버 `progress`는 단계마다 점프하므로, UI용 퍼센트를 부드럽게 올린다.
///
/// - 서버가 알려준 값(`confirmed`)을 넘지 않음 (완료 전)
/// - 폴링 사이에는 phase별 상한까지 천천히 증가 (inference 30→85 구간)
class AnalyzeProgressController extends ChangeNotifier {
  AnalyzeProgressController();

  static const Duration _tick = Duration(milliseconds: 50);

  Timer? _timer;
  double _confirmed = 0;
  double _visual = 0;
  static const String _defaultPhaseLabel = '서버로 전송 후 AI 분석 중입니다.';

  String _phaseLabel = _defaultPhaseLabel;
  String? _phase;
  String _status = 'queued';
  bool _terminal = false;
  bool _finishingDisplay = false;

  static const Duration settleAtComplete = Duration(milliseconds: 300);

  /// 0.0~1.0, UI 표시용
  double get visualProgress => _visual;

  int get visualPercent => (_visual * 100).round().clamp(0, 100);

  String get phaseLabel => _phaseLabel;

  void start() {
    _timer ??= Timer.periodic(_tick, (_) => _onTick());
  }

  void updateFromServer(AnalyzeJobStatus job) {
    _phase = job.phase;
    _status = job.status;
    _phaseLabel = job.phaseLabel;
    _confirmed = job.progress > _confirmed ? job.progress : _confirmed;
    if (job.isDone) {
      _confirmed = 1.0;
      _terminal = true;
    }
    if (job.isFailed) {
      _terminal = true;
    }
    notifyListeners();
  }

  /// 서버 분석이 끝난 뒤, 표시 진행률이 100%에 도달할 때까지 대기 후 [settleAtComplete]만큼 정지.
  Future<void> awaitVisualComplete({
    Duration? settleDelay,
    Duration timeout = const Duration(seconds: 20),
  }) async {
    _confirmed = 1.0;
    _terminal = true;
    _finishingDisplay = true;
    _phase = 'done';
    _phaseLabel = '완료';
    notifyListeners();

    final pause = settleDelay ?? settleAtComplete;
    final deadline = DateTime.now().add(timeout);

    while (_visual < 0.995 && DateTime.now().isBefore(deadline)) {
      await Future<void>.delayed(_tick);
    }

    _visual = 1.0;
    notifyListeners();
    await Future<void>.delayed(pause);
    _finishingDisplay = false;
  }

  void _onTick() {
    if (_finishingDisplay || (_terminal && _confirmed >= 1.0)) {
      final rate = _finishingDisplay ? 0.38 : 0.18;
      _visual = _easeToward(_visual, 1.0, rate: rate);
    } else {
      final ceiling = _effectiveCeiling(_phase, _status, _confirmed);
      double target;
      if (_visual < _confirmed - 0.008) {
        target = _confirmed;
      } else {
        target = (_visual + 0.0028).clamp(_confirmed, ceiling);
      }
      _visual = _easeToward(_visual, target, rate: 0.14);
    }
    _visual = _visual.clamp(0.0, 1.0);
    notifyListeners();
  }

  static double _effectiveCeiling(
    String? phase,
    String status,
    double confirmed,
  ) {
    return _phaseCeiling(phase, status)
        .clamp(confirmed, 0.99)
        .clamp(0.0, 0.99);
  }

  double _easeToward(double current, double target, {required double rate}) {
    final diff = target - current;
    if (diff.abs() < 0.001) return target;
    return current + diff * rate;
  }

  /// 서버 phase별 UI가 올라갈 수 있는 상한 (다음 단계 직전까지).
  static double _phaseCeiling(String? phase, String status) {
    switch (phase) {
      case 'upload':
        return 0.09;
      case 'fundus_check':
        return 0.24;
      case 'quickqual':
        return 0.29;
      case 'inference':
        return 0.84;
      case 'report':
        return 0.99;
      case 'done':
        return 1.0;
    }
    if (status == 'queued') return 0.06;
    if (status == 'running') return 0.92;
    return 0.95;
  }

  @override
  void dispose() {
    _timer?.cancel();
    _timer = null;
    super.dispose();
  }
}
