import 'dart:math' as math;

import 'package:flutter/material.dart';
import 'package:flutter/services.dart';

import '../api/eye_api_client.dart';
import '../config/api_config.dart';
import '../models/analysis_history_entry.dart';
import '../models/result_screen_args.dart';
import '../ui/medical_ui.dart';

/// 판정 필터 — [`_HistoryJudgmentIcon`]·목록 배지와 같은 기준.
enum _JudgmentFilter { all, abnormal, normal, unknown }

/// 기간 필터 (로컬 `createdAt` 기준).
enum _PeriodFilter { all, today, last7, last30, custom }

_JudgmentKind _judgmentKind(AnalysisHistoryEntry e) {
  final label = e.response.label ?? '';
  final lower = label.toLowerCase();
  if (lower.contains('abnormal') || label.contains('이상')) {
    return _JudgmentKind.abnormal;
  }
  if (label.isEmpty ||
      label.contains('판정 없음') ||
      lower.contains('unknown')) {
    return _JudgmentKind.unknown;
  }
  return _JudgmentKind.normal;
}

enum _JudgmentKind { abnormal, normal, unknown }

class _FilterDismissIntent extends Intent {
  const _FilterDismissIntent();
}

class _FilterConfirmIntent extends Intent {
  const _FilterConfirmIntent();
}

/// 이력 보기 — 백엔드 `GET /history` 에 저장된 영구 이력을 표시.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  HistoryViewMode _viewMode = HistoryViewMode.dashboard;

  bool _selectionMode = false;

  final Set<int> _selectedIndices = <int>{};

  final EyeApiClient _api = EyeApiClient();

  static const int _pageLimit = 40;

  List<AnalysisHistoryEntry> _entries = [];
  int _total = 0;
  bool _loadingRefresh = true;
  bool _loadingMore = false;

  bool _loadMoreBusy = false;
  String? _fetchError;

  bool _filterPanelOpen = false;
  _JudgmentFilter _judgmentFilter = _JudgmentFilter.all;
  _PeriodFilter _periodFilter = _PeriodFilter.all;
  DateTime? _customRangeStart;
  DateTime? _customRangeEnd;

  final LayerLink _filterLayerLink = LayerLink();
  final FocusNode _filterOverlayFocusNode = FocusNode();
  final GlobalKey<_CompactDateRangeCalendarState> _dateRangeCalendarKey =
      GlobalKey<_CompactDateRangeCalendarState>();
  OverlayEntry? _filterOverlayEntry;
  bool _calendarOpen = false;

  static const double _kFilterPanelWidth = 504;

  bool get _hasMore => _entries.length < _total;

  bool get _filtersActive {
    if (_judgmentFilter != _JudgmentFilter.all) return true;
    switch (_periodFilter) {
      case _PeriodFilter.all:
        return false;
      case _PeriodFilter.custom:
        return _customRangeStart != null && _customRangeEnd != null;
      case _PeriodFilter.today:
      case _PeriodFilter.last7:
      case _PeriodFilter.last30:
        return true;
    }
  }

  List<AnalysisHistoryEntry> get _filteredEntries {
    return _entries.where(_passesFilters).toList();
  }

  bool _passesFilters(AnalysisHistoryEntry e) {
    if (!_matchesJudgment(e)) return false;
    if (!_matchesPeriod(e)) return false;
    return true;
  }

  bool _matchesJudgment(AnalysisHistoryEntry e) {
    switch (_judgmentFilter) {
      case _JudgmentFilter.all:
        return true;
      case _JudgmentFilter.abnormal:
        return _judgmentKind(e) == _JudgmentKind.abnormal;
      case _JudgmentFilter.normal:
        return _judgmentKind(e) == _JudgmentKind.normal;
      case _JudgmentFilter.unknown:
        return _judgmentKind(e) == _JudgmentKind.unknown;
    }
  }

  bool _matchesPeriod(AnalysisHistoryEntry e) {
    final t = e.createdAt.toLocal();
    final now = DateTime.now();
    switch (_periodFilter) {
      case _PeriodFilter.all:
        return true;
      case _PeriodFilter.today:
        final d = DateTime(t.year, t.month, t.day);
        final n = DateTime(now.year, now.month, now.day);
        return d == n;
      case _PeriodFilter.last7:
        final today = DateTime(now.year, now.month, now.day);
        final from = today.subtract(const Duration(days: 7));
        final d = DateTime(t.year, t.month, t.day);
        return !d.isBefore(from);
      case _PeriodFilter.last30:
        final today = DateTime(now.year, now.month, now.day);
        final from = today.subtract(const Duration(days: 30));
        final d = DateTime(t.year, t.month, t.day);
        return !d.isBefore(from);
      case _PeriodFilter.custom:
        final s = _customRangeStart;
        final end = _customRangeEnd;
        // 확정된 날짜 범위가 없으면(선택 중·미적용) 목록을 비우지 않음
        if (s == null || end == null) return true;
        final day = DateTime(t.year, t.month, t.day);
        final sDay = DateTime(s.year, s.month, s.day);
        final eDay = DateTime(end.year, end.month, end.day);
        return !day.isBefore(sDay) && !day.isAfter(eDay);
    }
  }

  void _removeFilterOverlay() {
    _filterOverlayEntry?.remove();
    _filterOverlayEntry = null;
  }

  void _syncFilterOverlay() {
    _filterOverlayEntry?.markNeedsBuild();
  }

  void _showFilterOverlay() {
    if (!_filterPanelOpen || !mounted) return;
    _removeFilterOverlay();
    _filterOverlayEntry = OverlayEntry(
      builder: (_) => _buildFilterOverlayLayer(),
    );
    Overlay.of(context).insert(_filterOverlayEntry!);
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (mounted) {
        _filterOverlayFocusNode.requestFocus();
      }
    });
  }

  void _closeFilterPanelOverlay() {
    if (!mounted) return;
    setState(() {
      _filterPanelOpen = false;
      _calendarOpen = false;
      _selectedIndices.clear();
    });
    _removeFilterOverlay();
  }

  /// 하단 [확인] — 날짜 달력이 열려 있으면 선택 구간을 저장한 뒤 패널을 닫습니다.
  void _confirmFilterFooter() {
    if (!mounted) return;
    setState(() {
      if (_calendarOpen) {
        final r = _dateRangeCalendarKey.currentState?.commitPendingRange();
        if (r != null) {
          _periodFilter = _PeriodFilter.custom;
          _customRangeStart =
              DateTime(r.start.year, r.start.month, r.start.day);
          _customRangeEnd = DateTime(r.end.year, r.end.month, r.end.day);
        }
        _calendarOpen = false;
      }
      _filterPanelOpen = false;
      _selectedIndices.clear();
    });
    _removeFilterOverlay();
  }

  Widget _buildFilterOverlayLayer() {
    void barrierDismiss() => _closeFilterPanelOverlay();

    final stack = Stack(
      children: [
        Positioned.fill(
          child: GestureDetector(
            behavior: HitTestBehavior.opaque,
            onTap: barrierDismiss,
            child: ColoredBox(color: Colors.black.withValues(alpha: 0.28)),
          ),
        ),
        CompositedTransformFollower(
          link: _filterLayerLink,
          showWhenUnlinked: false,
          targetAnchor: Alignment.bottomRight,
          followerAnchor: Alignment.topRight,
          offset: const Offset(0, 6),
          child: Builder(
            builder: (overlayContext) {
              final screenW = MediaQuery.sizeOf(overlayContext).width;
              final horizontalMargin = 12.0;
              final panelW = math.min(
                _kFilterPanelWidth,
                math.max(280.0, screenW - horizontalMargin * 2),
              );
              return SizedBox(
                width: panelW,
                child: Material(
                  elevation: 12,
                  shadowColor: const Color(0x260C2A4A),
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
                  clipBehavior: Clip.antiAlias,
                  color: Colors.white,
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.stretch,
                    children: [
                      _HistoryFilterPanel(
                        judgment: _judgmentFilter,
                        period: _periodFilter,
                        customStart: _customRangeStart,
                        customEnd: _customRangeEnd,
                        onJudgmentChanged: (v) {
                          setState(() {
                            _judgmentFilter = v;
                            _selectedIndices.clear();
                          });
                          _syncFilterOverlay();
                        },
                        onPeriodChanged: (v) {
                          setState(() {
                            _periodFilter = v;
                            if (v != _PeriodFilter.custom) {
                              _customRangeStart = null;
                              _customRangeEnd = null;
                              _calendarOpen = false;
                            }
                            _selectedIndices.clear();
                          });
                          _syncFilterOverlay();
                        },
                        onDateRangeChipTap: () {
                          setState(() {
                            _periodFilter = _PeriodFilter.custom;
                            _calendarOpen = true;
                          });
                          _syncFilterOverlay();
                        },
                      ),
                      if (_calendarOpen)
                        _CompactDateRangeCalendar(
                          key: _dateRangeCalendarKey,
                          initialStart: _customRangeStart,
                          initialEnd: _customRangeEnd,
                        ),
                      const Divider(height: 1),
                      Padding(
                        padding: const EdgeInsets.symmetric(
                          horizontal: MedicalTokens.spaceSm,
                          vertical: 10,
                        ),
                        child: Row(
                          children: [
                            TextButton(
                              onPressed: _confirmFilterFooter,
                              child: const Text('확인'),
                            ),
                            const Spacer(),
                            TextButton(
                              onPressed: () {
                                setState(() {
                                  _resetFilters();
                                  _selectedIndices.clear();
                                });
                                _syncFilterOverlay();
                              },
                              child: const Text('초기화'),
                            ),
                          ],
                        ),
                      ),
                    ],
                  ),
                ),
              );
            },
          ),
        ),
      ],
    );

    return Shortcuts(
      shortcuts: <ShortcutActivator, Intent>{
        const SingleActivator(LogicalKeyboardKey.escape): const _FilterDismissIntent(),
        const SingleActivator(LogicalKeyboardKey.enter): const _FilterConfirmIntent(),
        const SingleActivator(LogicalKeyboardKey.numpadEnter): const _FilterConfirmIntent(),
      },
      child: Actions(
        actions: <Type, Action<Intent>>{
          _FilterDismissIntent: CallbackAction<_FilterDismissIntent>(
            onInvoke: (_) {
              barrierDismiss();
              return null;
            },
          ),
          _FilterConfirmIntent: CallbackAction<_FilterConfirmIntent>(
            onInvoke: (_) {
              _confirmFilterFooter();
              return null;
            },
          ),
        },
        child: Focus(
          autofocus: true,
          focusNode: _filterOverlayFocusNode,
          child: stack,
        ),
      ),
    );
  }

  void _exitSelectionMode() {
    setState(() {
      _selectionMode = false;
      _selectedIndices.clear();
    });
  }

  void _resetFilters() {
    setState(() {
      _judgmentFilter = _JudgmentFilter.all;
      _periodFilter = _PeriodFilter.all;
      _customRangeStart = null;
      _customRangeEnd = null;
      _calendarOpen = false;
    });
  }

  Future<void> _loadFirstPage() async {
    setState(() {
      _loadingRefresh = true;
      _fetchError = null;
      _selectionMode = false;
      _selectedIndices.clear();
    });

    try {
      final page = await _api.fetchHistoryPage(limit: _pageLimit, offset: 0);
      if (!mounted) return;
      setState(() {
        _entries = List<AnalysisHistoryEntry>.of(page.items);
        _total = page.total;
        _loadingRefresh = false;
        _loadingMore = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() {
        _fetchError = e.toString();
        _loadingRefresh = false;
      });
    }
  }

  Future<void> _loadMore() async {
    if (_loadMoreBusy || _loadingMore || !_hasMore || _loadingRefresh) return;

    _loadMoreBusy = true;
    setState(() => _loadingMore = true);
    try {
      final offset = _entries.length;
      final page = await _api.fetchHistoryPage(limit: _pageLimit, offset: offset);
      if (!mounted) return;
      setState(() {
        _entries.addAll(page.items);
        _total = page.total;
        _loadingMore = false;
      });
    } catch (e) {
      if (!mounted) return;
      setState(() => _loadingMore = false);
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('추가 로드 실패: $e')),
        );
      }
    } finally {
      _loadMoreBusy = false;
    }
  }

  bool _maybeTriggerLoadMore(ScrollMetrics metrics) {
    if (!_hasMore || _loadingMore) return false;
    if (metrics.pixels <= 0 || !metrics.hasPixels) return false;
    final nearEnd =
        metrics.pixels >= metrics.maxScrollExtent - 280;
    return nearEnd;
  }

  Future<void> _onTrashPressed() async {
    if (_entries.isEmpty || _loadingRefresh) return;

    if (!_selectionMode) {
      setState(() => _selectionMode = true);
      return;
    }

    await _deleteSelected();
  }

  @override
  void initState() {
    super.initState();
    _loadFirstPage();
  }

  @override
  void dispose() {
    _removeFilterOverlay();
    _filterOverlayFocusNode.dispose();
    _api.close();
    super.dispose();
  }

  void _syncSelectionToEntryCount(int length) {
    _selectedIndices.removeWhere((i) => i < 0 || i >= length);
  }

  void _toggleIndex(int index) {
    setState(() {
      if (_selectedIndices.contains(index)) {
        _selectedIndices.remove(index);
      } else {
        _selectedIndices.add(index);
      }
    });
  }

  void _toggleSelectAll() {
    setState(() {
      if (_filteredEntries.isEmpty) return;
      if (_selectedIndices.length == _filteredEntries.length) {
        _selectedIndices.clear();
      } else {
        _selectedIndices
          ..clear()
          ..addAll(List<int>.generate(_filteredEntries.length, (i) => i));
      }
    });
  }

  Future<void> _deleteSelected() async {
    if (_entries.isEmpty || _loadingRefresh) return;

    if (_selectedIndices.isEmpty) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('삭제할 항목을 선택해주세요.')),
      );
      return;
    }

    final sortedSnap = List<int>.from(_selectedIndices)..sort();
    final recordIds =
        sortedSnap.map((i) => _filteredEntries[i].recordId).toSet().toList();
    final count = recordIds.length;

    final confirmed = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('선택 항목 삭제'),
            content: Text('선택한 $count건의 이력을 삭제할까요?\n삭제 후에는 되돌릴 수 없습니다.'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('삭제'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('취소'),
              ),
            ],
          ),
        ) ??
        false;

    if (!confirmed || !mounted) return;

    try {
      for (final id in recordIds) {
        await _api.deleteHistoryRecord(id);
      }
      if (!mounted) return;

      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('선택한 이력을 삭제했습니다.')),
      );

      _exitSelectionMode();
      await _loadFirstPage();
    } catch (e) {
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('삭제 요청 실패: $e')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    _syncSelectionToEntryCount(_filteredEntries.length);

    final allSelected = _filteredEntries.isNotEmpty &&
        _selectedIndices.length == _filteredEntries.length;

    final appReady = !_loadingRefresh;

    late final Widget bodyContent;
    if (_loadingRefresh && _entries.isEmpty) {
      bodyContent = const Center(child: CircularProgressIndicator());
    } else if (_fetchError != null && _entries.isEmpty) {
      bodyContent = Center(
        child: Padding(
          padding: const EdgeInsets.all(MedicalTokens.spaceLg),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              Text(
                '이력을 불러오지 못했습니다.\n$_fetchError',
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: MedicalTokens.spaceMd),
              MedicalPrimaryButton(
                label: '다시 시도',
                onPressed: _loadFirstPage,
              ),
            ],
          ),
        ),
      );
    } else if (_entries.isEmpty) {
      bodyContent = const Center(
        child: Padding(
          padding: EdgeInsets.all(MedicalTokens.spaceLg),
          child: Text(
            '저장된 분석 이력이 없습니다.\n분석 후 이 화면을 새로고침하면 기록이 나타납니다.',
            textAlign: TextAlign.center,
          ),
        ),
      );
    } else if (_filteredEntries.isEmpty) {
      bodyContent = LayoutBuilder(
        builder: (context, constraints) {
          return RefreshIndicator(
            onRefresh: _loadFirstPage,
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              child: ConstrainedBox(
                constraints: BoxConstraints(minHeight: constraints.maxHeight),
                child: Center(
                  child: Padding(
                    padding: const EdgeInsets.all(MedicalTokens.spaceLg),
                    child: Column(
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        Text(
                          '조건에 맞는 이력이 없습니다.\n필터를 변경하거나 초기화해 보세요.',
                          textAlign: TextAlign.center,
                          style: Theme.of(context).textTheme.bodyLarge,
                        ),
                        const SizedBox(height: MedicalTokens.spaceMd),
                        Center(
                          child: Padding(
                            padding: const EdgeInsets.symmetric(horizontal: 24),
                            child: IntrinsicWidth(
                              child: MedicalSecondaryButton(
                                label: '필터 초기화',
                                onPressed: () {
                                  setState(_resetFilters);
                                },
                              ),
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          );
        },
      );
    } else {
      final scrollBody = NotificationListener<ScrollNotification>(
        onNotification: (n) {
          if (_maybeTriggerLoadMore(n.metrics)) {
            _loadMore();
          }
          return false;
        },
        child: _viewMode == HistoryViewMode.list
            ? _HistoryListView(
                entries: _filteredEntries,
                selectionMode: _selectionMode,
                selectedIndices: _selectedIndices,
                onToggleSelection: _toggleIndex,
              )
            : _HistoryDashboardView(
                entries: _filteredEntries,
                selectionMode: _selectionMode,
                selectedIndices: _selectedIndices,
                onToggleSelection: _toggleIndex,
              ),
      );
      bodyContent = RefreshIndicator(
        onRefresh: _loadFirstPage,
        child: scrollBody,
      );
    }

    final canPop = Navigator.of(context).canPop();

    return Scaffold(
      appBar: AppBar(
        actionsIconTheme: IconThemeData(
          color: MedicalTokens.textMain,
          size: 22,
        ),
        leading: canPop
            ? IconButton(
                icon: Icon(
                  Theme.of(context).platform == TargetPlatform.iOS
                      ? Icons.arrow_back_ios
                      : Icons.arrow_back,
                ),
                tooltip: MaterialLocalizations.of(context).backButtonTooltip,
                onPressed: () => Navigator.maybePop(context),
              )
            : null,
        automaticallyImplyLeading: false,
        centerTitle: true,
        title: _HistoryViewModeToggle(
          value: _viewMode,
          onChanged: (mode) {
            if (_viewMode == mode) return;
            setState(() => _viewMode = mode);
          },
        ),
        actions: [
          CompositedTransformTarget(
            link: _filterLayerLink,
            child: IconButton(
              tooltip: '필터',
              onPressed: !appReady
                  ? null
                  : () {
                      if (_entries.isEmpty && !_filterPanelOpen) return;
                      final opening = !_filterPanelOpen;
                      setState(() {
                        _filterPanelOpen = opening;
                        if (!opening) {
                          _calendarOpen = false;
                          _selectedIndices.clear();
                          _removeFilterOverlay();
                        }
                      });
                      if (opening && _entries.isNotEmpty) {
                        WidgetsBinding.instance.addPostFrameCallback((_) {
                          if (mounted) _showFilterOverlay();
                        });
                      }
                    },
              icon: _TuneSlidersGlyph(
                size: 22,
                color: !appReady
                    ? MedicalTokens.textSubtle
                    : (_filterPanelOpen || _filtersActive
                        ? Theme.of(context).colorScheme.primary
                        : MedicalTokens.textMain),
              ),
            ),
          ),
          IconButton(
            onPressed: (_entries.isEmpty || !appReady) ? null : _onTrashPressed,
            icon: const Icon(Icons.delete_outline),
            tooltip: _selectionMode ? '선택 항목 삭제' : '삭제할 항목 선택',
          ),
          if (_selectionMode && _entries.isNotEmpty && appReady) ...[
            IconButton(
              onPressed: _filteredEntries.isEmpty ? null : _toggleSelectAll,
              icon: Icon(allSelected ? Icons.deselect : Icons.select_all),
              tooltip: allSelected ? '전체 선택 해제' : '전체 선택',
            ),
            IconButton(
              onPressed: _exitSelectionMode,
              icon: const Icon(Icons.close),
              tooltip: '선택 종료',
            ),
          ],
        ],
      ),
      body: Stack(
        children: [
          Positioned.fill(child: bodyContent),
          if (_loadingMore)
            Positioned(
              left: 0,
              right: 0,
              bottom: 16,
              child: Center(
                child: Material(
                  borderRadius: BorderRadius.circular(20),
                  color: Colors.white,
                  elevation: 2,
                  child: const Padding(
                    padding: EdgeInsets.symmetric(horizontal: 16, vertical: 10),
                    child: SizedBox(
                      height: 20,
                      width: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  ),
                ),
              ),
            ),
        ],
      ),
    );
  }
}

/// Material 아이콘 폰트가 웹 빌드 등에서 로드되지 않을 때를 대비해, 튜닝(가로 슬라이더)을 직접 그립니다.
class _TuneSlidersGlyph extends StatelessWidget {
  const _TuneSlidersGlyph({
    required this.color,
    this.size = 22,
  });

  final Color color;
  final double size;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      size: Size.square(size),
      painter: _TuneSlidersPainter(color),
    );
  }
}

class _TuneSlidersPainter extends CustomPainter {
  _TuneSlidersPainter(this.color);

  final Color color;

  @override
  void paint(Canvas canvas, Size size) {
    final w = size.width;
    final h = size.height;
    final trackW = math.max(1.2, h * 0.075);
    final knobR = math.max(2.2, h * 0.11);

    final trackPaint = Paint()
      ..color = color.withValues(alpha: 0.45)
      ..strokeWidth = trackW
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;

    final knobPaint = Paint()
      ..color = color
      ..style = PaintingStyle.fill;

    final pad = w * 0.12;
    final xL = pad;
    final xR = w - pad;
    final ys = [h * 0.26, h * 0.5, h * 0.74];
    final tKnob = [0.38, 0.66, 0.48];

    for (var i = 0; i < 3; i++) {
      final y = ys[i];
      canvas.drawLine(Offset(xL, y), Offset(xR, y), trackPaint);
      final span = (xR - xL);
      var kx = xL + span * tKnob[i];
      final lo = xL + knobR;
      final hi = xR - knobR;
      if (kx < lo) kx = lo;
      if (kx > hi) kx = hi;
      canvas.drawCircle(Offset(kx, y), knobR, knobPaint);
    }
  }

  @override
  bool shouldRepaint(covariant _TuneSlidersPainter oldDelegate) =>
      oldDelegate.color != color;
}

/// 한 달만 표시하고 화살표로 월 이동하는 컴팩트 기간 선택(필터 패널 아래 오버레이용).
class _CompactDateRangeCalendar extends StatefulWidget {
  const _CompactDateRangeCalendar({
    super.key,
    required this.initialStart,
    required this.initialEnd,
  });

  final DateTime? initialStart;
  final DateTime? initialEnd;

  @override
  State<_CompactDateRangeCalendar> createState() =>
      _CompactDateRangeCalendarState();
}

class _CompactDateRangeCalendarState extends State<_CompactDateRangeCalendar> {
  static DateTime _onlyDay(DateTime d) => DateTime(d.year, d.month, d.day);

  late DateTime _monthPage;
  DateTime? _pickStart;
  DateTime? _pickEnd;

  DateTime get _firstDate => DateTime(2000, 1, 1);
  DateTime get _lastDate => DateTime(DateTime.now().year + 1, 12, 31);

  @override
  void initState() {
    super.initState();
    _pickStart = widget.initialStart != null ? _onlyDay(widget.initialStart!) : null;
    _pickEnd = widget.initialEnd != null ? _onlyDay(widget.initialEnd!) : null;
    final anchor = _pickStart ?? _pickEnd ?? DateTime.now();
    _monthPage = DateTime(anchor.year, anchor.month, 1);
    _clampMonth();
  }

  void _clampMonth() {
    final minM = DateTime(_firstDate.year, _firstDate.month, 1);
    final maxM = DateTime(_lastDate.year, _lastDate.month, 1);
    if (_monthPage.isBefore(minM)) _monthPage = minM;
    if (_monthPage.isAfter(maxM)) _monthPage = maxM;
  }

  bool _canGoPrev() {
    final prev = DateTime(_monthPage.year, _monthPage.month - 1, 1);
    return !prev.isBefore(DateTime(_firstDate.year, _firstDate.month, 1));
  }

  bool _canGoNext() {
    final next = DateTime(_monthPage.year, _monthPage.month + 1, 1);
    return !next.isAfter(DateTime(_lastDate.year, _lastDate.month, 1));
  }

  void _prevMonth() {
    if (!_canGoPrev()) return;
    setState(() {
      _monthPage = DateTime(_monthPage.year, _monthPage.month - 1, 1);
      _clampMonth();
    });
  }

  void _nextMonth() {
    if (!_canGoNext()) return;
    setState(() {
      _monthPage = DateTime(_monthPage.year, _monthPage.month + 1, 1);
      _clampMonth();
    });
  }

  void _onDayTap(DateTime raw) {
    final day = _onlyDay(raw);
    if (day.isBefore(_onlyDay(_firstDate)) || day.isAfter(_onlyDay(_lastDate))) {
      return;
    }
    setState(() {
      if (_pickStart == null || (_pickStart != null && _pickEnd != null)) {
        _pickStart = day;
        _pickEnd = null;
      } else {
        if (day.isBefore(_pickStart!)) {
          _pickEnd = _pickStart;
          _pickStart = day;
        } else {
          _pickEnd = day;
        }
      }
    });
  }

  /// 패널 [확인] 시 선택 구간을 부모에 넘기기 위해 호출합니다.
  DateTimeRange? commitPendingRange() {
    if (_pickStart == null) return null;
    final start = _onlyDay(_pickStart!);
    final end = _onlyDay(_pickEnd ?? _pickStart!);
    return DateTimeRange(start: start, end: end);
  }

  bool _sameDay(DateTime a, DateTime b) =>
      a.year == b.year && a.month == b.month && a.day == b.day;

  bool _inSelectedRange(DateTime day) {
    if (_pickStart == null || _pickEnd == null) return false;
    final d = _onlyDay(day);
    return !d.isBefore(_pickStart!) && !d.isAfter(_pickEnd!);
  }

  String _monthTitle(MaterialLocalizations loc) {
    return loc.formatMonthYear(_monthPage);
  }

  @override
  Widget build(BuildContext context) {
    final loc = MaterialLocalizations.of(context);
    final cs = Theme.of(context).colorScheme;
    final wdLabels = loc.narrowWeekdays;
    final startIdx = loc.firstDayOfWeekIndex;

    final daysInMonth = DateTime(_monthPage.year, _monthPage.month + 1, 0).day;
    final leading = DateUtils.firstDayOffset(
      _monthPage.year,
      _monthPage.month,
      loc,
    );

    final cells = <Widget?>[];
    for (var i = 0; i < leading; i++) {
      cells.add(null);
    }
    for (var d = 1; d <= daysInMonth; d++) {
      final date = DateTime(_monthPage.year, _monthPage.month, d);
      cells.add(_dayCell(context, date, cs));
    }
    while (cells.length % 7 != 0) {
      cells.add(null);
    }

    return Padding(
      padding: const EdgeInsets.fromLTRB(12, 0, 12, 12),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          const Divider(height: 1),
          const SizedBox(height: 10),
          Row(
            children: [
              IconButton(
                onPressed: _canGoPrev() ? _prevMonth : null,
                icon: const Icon(Icons.chevron_left),
                tooltip: '이전 달',
              ),
              Expanded(
                child: Text(
                  _monthTitle(loc),
                  textAlign: TextAlign.center,
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.w800,
                        color: MedicalTokens.textMain,
                      ),
                ),
              ),
              IconButton(
                onPressed: _canGoNext() ? _nextMonth : null,
                icon: const Icon(Icons.chevron_right),
                tooltip: '다음 달',
              ),
            ],
          ),
          const SizedBox(height: 6),
          Row(
            children: List.generate(7, (i) {
              final idx = (startIdx + i) % 7;
              return Expanded(
                child: Center(
                  child: Text(
                    wdLabels[idx],
                    style: Theme.of(context).textTheme.labelSmall?.copyWith(
                          color: MedicalTokens.textSubtle,
                          fontWeight: FontWeight.w600,
                        ),
                  ),
                ),
              );
            }),
          ),
          const SizedBox(height: 6),
          ...List.generate(cells.length ~/ 7, (row) {
            return Padding(
              padding: const EdgeInsets.only(bottom: 4),
              child: Row(
                children: List.generate(7, (col) {
                  final idx = row * 7 + col;
                  final w = cells[idx];
                  return Expanded(child: w ?? const SizedBox(height: 36));
                }),
              ),
            );
          }),
        ],
      ),
    );
  }

  Widget _dayCell(
    BuildContext context,
    DateTime date,
    ColorScheme cs,
  ) {
    final dayOnly = _onlyDay(date);
    if (dayOnly.isBefore(_onlyDay(_firstDate)) || dayOnly.isAfter(_onlyDay(_lastDate))) {
      return const SizedBox(height: 36);
    }

    final label = '${date.day}';
    final isStart = _pickStart != null && _sameDay(date, _pickStart!);
    final isEnd = _pickEnd != null && _sameDay(date, _pickEnd!);
    final inRange = _inSelectedRange(date);
    final highlight = isStart || isEnd || inRange;

    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 1),
      child: InkWell(
        onTap: () => _onDayTap(date),
        borderRadius: BorderRadius.circular(8),
        child: DecoratedBox(
          decoration: BoxDecoration(
            color: highlight ? MedicalTokens.primarySoft : Colors.transparent,
            borderRadius: BorderRadius.circular(8),
            border: isStart || isEnd
                ? Border.all(color: cs.primary, width: 1.5)
                : null,
          ),
          child: SizedBox(
            height: 36,
            child: Center(
              child: Text(
                label,
                style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                      fontWeight: isStart || isEnd ? FontWeight.w800 : FontWeight.w500,
                      color: MedicalTokens.textMain,
                    ),
              ),
            ),
          ),
        ),
      ),
    );
  }
}

class _HistoryFilterPanel extends StatelessWidget {
  const _HistoryFilterPanel({
    required this.judgment,
    required this.period,
    required this.customStart,
    required this.customEnd,
    required this.onJudgmentChanged,
    required this.onPeriodChanged,
    required this.onDateRangeChipTap,
  });

  final _JudgmentFilter judgment;
  final _PeriodFilter period;
  final DateTime? customStart;
  final DateTime? customEnd;
  final ValueChanged<_JudgmentFilter> onJudgmentChanged;
  final ValueChanged<_PeriodFilter> onPeriodChanged;
  final VoidCallback onDateRangeChipTap;

  static String _dateLabel(DateTime d) =>
      '${d.year}-${d.month.toString().padLeft(2, '0')}-${d.day.toString().padLeft(2, '0')}';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final dateChipLabel = customStart != null && customEnd != null
        ? '${_dateLabel(customStart!)} ~ ${_dateLabel(customEnd!)}'
        : '날짜 범위 선택';

    return Padding(
      padding: const EdgeInsets.fromLTRB(
        MedicalTokens.spaceMd,
        MedicalTokens.spaceSm,
        MedicalTokens.spaceMd,
        MedicalTokens.spaceMd,
      ),
      child: Column(
        mainAxisSize: MainAxisSize.min,
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Row(
            children: [
              Text(
                '판정',
                style: theme.textTheme.titleSmall?.copyWith(
                  fontWeight: FontWeight.w800,
                  color: MedicalTokens.textMain,
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: [
              _HistoryFilterChip(
                label: '전체',
                selected: judgment == _JudgmentFilter.all,
                onTap: () => onJudgmentChanged(_JudgmentFilter.all),
              ),
              _HistoryFilterChip(
                label: '이상',
                selected: judgment == _JudgmentFilter.abnormal,
                onTap: () => onJudgmentChanged(_JudgmentFilter.abnormal),
              ),
              _HistoryFilterChip(
                label: '정상',
                selected: judgment == _JudgmentFilter.normal,
                onTap: () => onJudgmentChanged(_JudgmentFilter.normal),
              ),
              _HistoryFilterChip(
                label: '판정 없음',
                selected: judgment == _JudgmentFilter.unknown,
                onTap: () => onJudgmentChanged(_JudgmentFilter.unknown),
              ),
            ],
          ),
          const SizedBox(height: MedicalTokens.spaceMd),
          Text(
            '기간',
            style: theme.textTheme.titleSmall?.copyWith(
              fontWeight: FontWeight.w800,
              color: MedicalTokens.textMain,
            ),
          ),
          const SizedBox(height: 8),
          SingleChildScrollView(
            scrollDirection: Axis.horizontal,
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _HistoryFilterChip(
                  label: '전체',
                  dense: true,
                  selected: period == _PeriodFilter.all,
                  onTap: () => onPeriodChanged(_PeriodFilter.all),
                ),
                const SizedBox(width: 6),
                _HistoryFilterChip(
                  label: '오늘',
                  dense: true,
                  selected: period == _PeriodFilter.today,
                  onTap: () => onPeriodChanged(_PeriodFilter.today),
                ),
                const SizedBox(width: 6),
                _HistoryFilterChip(
                  label: '최근 7일',
                  dense: true,
                  selected: period == _PeriodFilter.last7,
                  onTap: () => onPeriodChanged(_PeriodFilter.last7),
                ),
                const SizedBox(width: 6),
                _HistoryFilterChip(
                  label: '최근 30일',
                  dense: true,
                  selected: period == _PeriodFilter.last30,
                  onTap: () => onPeriodChanged(_PeriodFilter.last30),
                ),
                const SizedBox(width: 6),
                _HistoryFilterChip(
                  label: dateChipLabel,
                  dense: true,
                  selected: period == _PeriodFilter.custom,
                  onTap: onDateRangeChipTap,
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _HistoryFilterChip extends StatelessWidget {
  const _HistoryFilterChip({
    required this.label,
    required this.selected,
    required this.onTap,
    this.dense = false,
  });

  final String label;
  final bool selected;
  final VoidCallback onTap;
  final bool dense;

  @override
  Widget build(BuildContext context) {
    final hPad = dense ? 9.0 : 14.0;
    final vPad = dense ? 7.0 : 8.0;
    return Material(
      color: selected ? MedicalTokens.primarySoft : const Color(0xFFF6F7FB),
      borderRadius: BorderRadius.circular(999),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(999),
        child: Padding(
          padding: EdgeInsets.symmetric(horizontal: hPad, vertical: vPad),
          child: Text(
            label,
            maxLines: 1,
            overflow: TextOverflow.ellipsis,
            style: Theme.of(context).textTheme.labelLarge?.copyWith(
                  fontWeight: FontWeight.w700,
                  fontSize: dense ? 12.5 : null,
                  color: selected
                      ? Theme.of(context).colorScheme.primary
                      : MedicalTokens.textMain,
                ),
          ),
        ),
      ),
    );
  }
}

enum HistoryViewMode { dashboard, list }

class _HistoryViewModeToggle extends StatelessWidget {
  const _HistoryViewModeToggle({
    required this.value,
    required this.onChanged,
  });

  final HistoryViewMode value;
  final ValueChanged<HistoryViewMode> onChanged;

  @override
  Widget build(BuildContext context) {
    final selectedColor = Theme.of(context).colorScheme.primary;
    final unselectedColor = Theme.of(context).colorScheme.onSurfaceVariant;
    final selectedBg = MedicalTokens.primarySoft;

    Widget button({
      required IconData icon,
      required HistoryViewMode mode,
      required String tooltip,
    }) {
      final selected = value == mode;
      return InkWell(
        borderRadius: BorderRadius.circular(10),
        onTap: () => onChanged(mode),
        child: Container(
          width: 40,
          height: 34,
          decoration: BoxDecoration(
            color: selected ? selectedBg : Colors.transparent,
            borderRadius: BorderRadius.circular(10),
          ),
          alignment: Alignment.center,
          child: Icon(
            icon,
            size: 20,
            color: selected ? selectedColor : unselectedColor,
          ),
        ),
      );
    }

    return DecoratedBox(
      decoration: BoxDecoration(
        color: const Color(0xFFF6F7FB),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: MedicalTokens.border),
      ),
      child: Padding(
        padding: const EdgeInsets.all(3),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Tooltip(
              message: '대시보드 보기',
              child: button(
                icon: Icons.grid_view_rounded,
                mode: HistoryViewMode.dashboard,
                tooltip: '대시보드 보기',
              ),
            ),
            const SizedBox(width: 4),
            Tooltip(
              message: '리스트 보기',
              child: button(
                icon: Icons.list_rounded,
                mode: HistoryViewMode.list,
                tooltip: '리스트 보기',
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _HistoryListView extends StatelessWidget {
  const _HistoryListView({
    required this.entries,
    required this.selectionMode,
    required this.selectedIndices,
    required this.onToggleSelection,
  });

  final List<AnalysisHistoryEntry> entries;
  final bool selectionMode;
  final Set<int> selectedIndices;
  final ValueChanged<int> onToggleSelection;

  @override
  Widget build(BuildContext context) {
    return ListView.separated(
      physics: const AlwaysScrollableScrollPhysics(),
      padding: const EdgeInsets.all(MedicalTokens.spaceMd),
      itemCount: entries.length,
      separatorBuilder: (_, _) => const SizedBox(height: MedicalTokens.spaceSm),
      itemBuilder: (context, index) {
        final item = entries[index];
        final subtitle = item.response.label ?? '판정 없음';
        final isAbnormal = subtitle.contains('abnormal');
        final selected = selectedIndices.contains(index);
        final rowContent = InkWell(
          onTap: selectionMode
              ? () => onToggleSelection(index)
              : () => _openResult(context, item),
          borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
          child: Padding(
            padding: EdgeInsets.fromLTRB(
              selectionMode ? 0 : MedicalTokens.spaceSm,
              8,
              MedicalTokens.spaceSm,
              8,
            ),
            child: Row(
              crossAxisAlignment: CrossAxisAlignment.center,
              children: [
                ClipRRect(
                  borderRadius: BorderRadius.circular(10),
                  child: DecoratedBox(
                    decoration: BoxDecoration(
                      border: Border.all(color: MedicalTokens.border),
                      borderRadius: BorderRadius.circular(10),
                    ),
                            child: _HistoryThumbnail(
                              entry: item,
                              width: 54,
                              height: 54,
                              boxFit: BoxFit.cover,
                            ),
                  ),
                ),
                const SizedBox(width: MedicalTokens.spaceSm),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        item.filename,
                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                              fontWeight: FontWeight.w800,
                            ),
                      ),
                      const SizedBox(height: 6),
                      Text(
                        _formatDateTime(item.createdAt),
                        style: Theme.of(context).textTheme.bodySmall?.copyWith(
                              color: Theme.of(context).colorScheme.onSurfaceVariant,
                            ),
                      ),
                    ],
                  ),
                ),
                Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.end,
                  children: [
                    MedicalBadge(
                      text: subtitle,
                      backgroundColor:
                          isAbnormal ? const Color(0xFFFFEEE8) : MedicalTokens.primarySoft,
                      foregroundColor:
                          isAbnormal ? const Color(0xFFC46235) : MedicalTokens.textMain,
                    ),
                    const SizedBox(height: 6),
                    const Icon(Icons.chevron_right, size: 18),
                  ],
                ),
              ],
            ),
          ),
        );

        return MedicalCard(
          padding: const EdgeInsets.symmetric(
            horizontal: MedicalTokens.spaceXs,
            vertical: MedicalTokens.spaceXs,
          ),
          child: Row(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              if (selectionMode)
                Checkbox(
                  value: selected,
                  visualDensity: VisualDensity.compact,
                  materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                  onChanged: (_) => onToggleSelection(index),
                ),
              Expanded(child: rowContent),
            ],
          ),
        );
      },
    );
  }
}

class _HistoryDashboardView extends StatelessWidget {
  const _HistoryDashboardView({
    required this.entries,
    required this.selectionMode,
    required this.selectedIndices,
    required this.onToggleSelection,
  });

  final List<AnalysisHistoryEntry> entries;
  final bool selectionMode;
  final Set<int> selectedIndices;
  final ValueChanged<int> onToggleSelection;

  /// 카드 하단 메타 바(이름·날짜) 고정 높이 — 그리드 `childAspectRatio` 계산과 맞춘다.
  static const double _metaBarHeight = 68;

  @override
  Widget build(BuildContext context) {
    return LayoutBuilder(
      builder: (context, constraints) {
        final width = constraints.maxWidth;
        final crossAxisCount = width >= 1200 ? 4 : (width >= 900 ? 3 : 2);

        final gridHorizontalPadding = MedicalTokens.spaceMd * 2;
        final innerCrossExtent = (width - gridHorizontalPadding).clamp(0.0, double.infinity);
        final crossAxisSpacingTotal = MedicalTokens.spaceSm * (crossAxisCount - 1);
        final cellCrossExtent =
            (innerCrossExtent - crossAxisSpacingTotal) / crossAxisCount;
        final cellMainExtent = cellCrossExtent + _metaBarHeight;
        final childAspectRatio = cellCrossExtent / cellMainExtent;

        return GridView.builder(
          physics: const AlwaysScrollableScrollPhysics(),
          padding: const EdgeInsets.all(MedicalTokens.spaceMd),
          gridDelegate: SliverGridDelegateWithFixedCrossAxisCount(
            crossAxisCount: crossAxisCount,
            crossAxisSpacing: MedicalTokens.spaceSm,
            mainAxisSpacing: MedicalTokens.spaceSm,
            childAspectRatio: childAspectRatio,
          ),
          itemCount: entries.length,
          itemBuilder: (context, index) {
            final item = entries[index];
            final selected =
                selectionMode && selectedIndices.contains(index);

            final card = InkWell(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
              onTap: selectionMode
                  ? () => onToggleSelection(index)
                  : () => _openResult(context, item),
              child: DecoratedBox(
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
                  border: Border.all(
                    color: selected ? Theme.of(context).colorScheme.primary : MedicalTokens.border,
                    width: selected ? 2 : 1,
                  ),
                ),
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(MedicalTokens.radiusLg - 1),
                  child: Padding(
                    padding: const EdgeInsets.all(1),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.stretch,
                      mainAxisSize: MainAxisSize.min,
                      children: [
                        AspectRatio(
                          aspectRatio: 1,
                          child: ClipRRect(
                            borderRadius: BorderRadius.vertical(
                              top: Radius.circular(MedicalTokens.radiusLg - 2),
                            ),
                            child: ColoredBox(
                              color: Colors.black,
                              child: _HistoryThumbnail(
                                entry: item,
                                boxFit: BoxFit.contain,
                              ),
                            ),
                          ),
                        ),
                        SizedBox(
                          height: _metaBarHeight,
                          child: Container(
                            width: double.infinity,
                            padding: const EdgeInsets.fromLTRB(10, 8, 10, 10),
                            decoration: const BoxDecoration(
                              color: Color(0xFFF8FAFD),
                            ),
                            child: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisAlignment: MainAxisAlignment.center,
                              children: [
                                Row(
                                  children: [
                                    Expanded(
                                      child: Text(
                                        item.filename,
                                        maxLines: 1,
                                        overflow: TextOverflow.ellipsis,
                                        style: Theme.of(context).textTheme.titleMedium?.copyWith(
                                              fontWeight: FontWeight.w800,
                                            ),
                                      ),
                                    ),
                                    const SizedBox(width: 6),
                                    _HistoryJudgmentIcon(label: item.response.label),
                                  ],
                                ),
                                const SizedBox(height: 4),
                                Text(
                                  _formatDateTime(item.createdAt),
                                  maxLines: 1,
                                  overflow: TextOverflow.ellipsis,
                                  style: Theme.of(context).textTheme.bodySmall?.copyWith(
                                        color:
                                            Theme.of(context).colorScheme.onSurfaceVariant,
                                      ),
                                ),
                              ],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            );

            if (!selectionMode) return card;

            return Stack(
              clipBehavior: Clip.none,
              children: [
                card,
                Positioned(
                  top: 6,
                  left: 6,
                  child: Material(
                    color: Colors.white.withValues(alpha: 0.92),
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(6),
                      side: BorderSide(color: MedicalTokens.border),
                    ),
                    clipBehavior: Clip.antiAlias,
                    child: InkWell(
                      onTap: () => onToggleSelection(index),
                      child: SizedBox(
                        width: 28,
                        height: 28,
                        child: Checkbox(
                          value: selectedIndices.contains(index),
                          materialTapTargetSize: MaterialTapTargetSize.shrinkWrap,
                          visualDensity: VisualDensity.compact,
                          onChanged: (_) => onToggleSelection(index),
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            );
          },
        );
      },
    );
  }
}

class _HistoryJudgmentIcon extends StatelessWidget {
  const _HistoryJudgmentIcon({required this.label});

  final String? label;

  @override
  Widget build(BuildContext context) {
    final raw = (label ?? '').toLowerCase();
    if (raw.contains('abnormal') || raw.contains('이상')) {
      return const Icon(Icons.warning_amber_rounded, size: 18, color: Color(0xFFC46235));
    }
    if (raw.isEmpty || raw.contains('판정 없음')) {
      return Icon(
        Icons.help_outline_rounded,
        size: 18,
        color: Theme.of(context).colorScheme.onSurfaceVariant,
      );
    }
    return const Icon(Icons.check_circle_rounded, size: 18, color: MedicalTokens.success);
  }
}

void _openResult(BuildContext context, AnalysisHistoryEntry item) {
  Navigator.pushNamed(
    context,
    '/result',
    arguments: ResultScreenArgs(
      analyzeResponse: item.response,
      originalImageBytes: item.originalImageBytes,
    ),
  );
}

class _HistoryThumbnail extends StatelessWidget {
  const _HistoryThumbnail({
    required this.entry,
    required this.boxFit,
    this.width,
    this.height,
  });

  final AnalysisHistoryEntry entry;
  final BoxFit boxFit;
  final double? width;
  final double? height;

  @override
  Widget build(BuildContext context) {
    Widget child;
    if (entry.originalImageBytes != null) {
      child = Image.memory(entry.originalImageBytes!, fit: boxFit);
    } else {
      final path = entry.response.originalUrl;
      if (path == null || path.isEmpty) {
        child = Center(
          child: Icon(
            Icons.hide_image_outlined,
            color: Theme.of(context).colorScheme.onSurfaceVariant,
          ),
        );
      } else {
        child = Image.network(
          ApiConfig.resolveAssetUrl(path),
          fit: boxFit,
          loadingBuilder: (context, widget, prog) =>
              prog == null ? widget : const Center(child: CircularProgressIndicator()),
          errorBuilder: (_, _, _) => Center(
            child: Icon(
              Icons.broken_image_outlined,
              color: Theme.of(context).colorScheme.onSurfaceVariant,
            ),
          ),
        );
      }
    }

    if (width != null || height != null) {
      return SizedBox(width: width, height: height, child: child);
    }
    return SizedBox.expand(child: child);
  }
}

String _formatDateTime(DateTime value) {
  final y = value.year.toString().padLeft(4, '0');
  final m = value.month.toString().padLeft(2, '0');
  final d = value.day.toString().padLeft(2, '0');
  final hh = value.hour.toString().padLeft(2, '0');
  final mm = value.minute.toString().padLeft(2, '0');
  return '$y-$m-$d $hh:$mm';
}
