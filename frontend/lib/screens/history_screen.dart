import 'package:flutter/material.dart';

import '../api/eye_api_client.dart';
import '../config/api_config.dart';
import '../models/analysis_history_entry.dart';
import '../models/result_screen_args.dart';
import '../ui/medical_ui.dart';

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

  bool get _hasMore => _entries.length < _total;

  void _exitSelectionMode() {
    setState(() {
      _selectionMode = false;
      _selectedIndices.clear();
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
      if (_entries.isEmpty) return;
      if (_selectedIndices.length == _entries.length) {
        _selectedIndices.clear();
      } else {
        _selectedIndices
          ..clear()
          ..addAll(List<int>.generate(_entries.length, (i) => i));
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
        sortedSnap.map((i) => _entries[i].recordId).toSet().toList();
    final count = recordIds.length;

    final confirmed = await showDialog<bool>(
          context: context,
          builder: (ctx) => AlertDialog(
            title: const Text('선택 항목 삭제'),
            content: Text('선택한 $count건의 이력을 삭제할까요?\n삭제 후에는 되돌릴 수 없습니다.'),
            actions: [
              TextButton(
                onPressed: () => Navigator.pop(ctx, false),
                child: const Text('취소'),
              ),
              TextButton(
                onPressed: () => Navigator.pop(ctx, true),
                child: const Text('삭제'),
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
    _syncSelectionToEntryCount(_entries.length);

    final allSelected =
        _entries.isNotEmpty && _selectedIndices.length == _entries.length;

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
                entries: _entries,
                selectionMode: _selectionMode,
                selectedIndices: _selectedIndices,
                onToggleSelection: _toggleIndex,
              )
            : _HistoryDashboardView(
                entries: _entries,
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
          IconButton(
            onPressed: (_entries.isEmpty || !appReady) ? null : _onTrashPressed,
            icon: const Icon(Icons.delete_outline),
            tooltip: _selectionMode ? '선택 항목 삭제' : '삭제할 항목 선택',
          ),
          if (_selectionMode && _entries.isNotEmpty && appReady) ...[
            IconButton(
              onPressed: _toggleSelectAll,
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
          onTap: () => _openResult(context, item),
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

  static const Color _galleryImageBg = Color(0xFFF2F5F9);

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
              onTap: () => _openResult(context, item),
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
                              color: _galleryImageBg,
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
          errorBuilder: (_, __, ___) => Center(
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
