import 'package:flutter/material.dart';

import '../models/result_screen_args.dart';
import '../state/analysis_history_store.dart';
import '../ui/medical_ui.dart';

/// 이력 보기 — 현재 세션에서 `/analyze` 성공 응답을 저장해 표시.
class HistoryScreen extends StatefulWidget {
  const HistoryScreen({super.key});

  @override
  State<HistoryScreen> createState() => _HistoryScreenState();
}

class _HistoryScreenState extends State<HistoryScreen> {
  void _clearHistory() {
    AnalysisHistoryStore.clear();
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    final entries = AnalysisHistoryStore.entries;

    return Scaffold(
      appBar: AppBar(
        title: const Text('분석 이력'),
        actions: [
          IconButton(
            onPressed: entries.isEmpty ? null : _clearHistory,
            icon: const Icon(Icons.delete_outline),
            tooltip: '이력 비우기',
          ),
        ],
      ),
      body: entries.isEmpty
          ? const Center(
              child: Padding(
                padding: EdgeInsets.all(MedicalTokens.spaceLg),
                child: Text(
                  '현재 세션에 분석 이력이 없습니다.\n이미지를 업로드해 분석을 먼저 진행해주세요.',
                  textAlign: TextAlign.center,
                ),
              ),
            )
          : ListView.separated(
              padding: const EdgeInsets.all(MedicalTokens.spaceMd),
              itemCount: entries.length,
              separatorBuilder: (_, _) => const SizedBox(height: MedicalTokens.spaceSm),
              itemBuilder: (context, index) {
                final item = entries[index];
                final subtitle = item.response.label ?? '판정 없음';
                final isAbnormal = subtitle.contains('abnormal');
                return MedicalCard(
                  padding: const EdgeInsets.symmetric(
                    horizontal: MedicalTokens.spaceSm,
                    vertical: MedicalTokens.spaceXs,
                  ),
                  child: ListTile(
                    contentPadding: const EdgeInsets.symmetric(horizontal: 4),
                    leading: ClipRRect(
                      borderRadius: BorderRadius.circular(10),
                      child: DecoratedBox(
                        decoration: BoxDecoration(
                          border: Border.all(color: MedicalTokens.border),
                          borderRadius: BorderRadius.circular(10),
                        ),
                        child: Image.memory(
                          item.originalImageBytes,
                          width: 54,
                          height: 54,
                          fit: BoxFit.cover,
                        ),
                      ),
                    ),
                    title: Text(item.filename),
                    subtitle: Padding(
                      padding: const EdgeInsets.only(top: 6),
                      child: Text(_formatDateTime(item.createdAt)),
                    ),
                    trailing: Column(
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
                    onTap: () {
                      Navigator.pushNamed(
                        context,
                        '/result',
                        arguments: ResultScreenArgs(
                          originalImageBytes: item.originalImageBytes,
                          analyzeResponse: item.response,
                        ),
                      );
                    },
                  ),
                );
              },
            ),
    );
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
