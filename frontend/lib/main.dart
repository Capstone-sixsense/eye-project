import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'models/result_screen_args.dart';
import 'screens/history_screen.dart';
import 'screens/result_screen.dart';
import 'screens/upload_screen.dart';
import 'ui/medical_ui.dart';

void main() {
  final colorScheme = ColorScheme.fromSeed(
    seedColor: MedicalTokens.primary,
    brightness: Brightness.light,
  );

  final baseText = Typography.material2021().black.apply(
    fontFamilyFallback: const ['Pretendard', 'Noto Sans KR', 'Apple SD Gothic Neo'],
  );

  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        useMaterial3: true,
        colorScheme: colorScheme,
        scaffoldBackgroundColor: MedicalTokens.background,
        textTheme: baseText.copyWith(
          displaySmall: baseText.displaySmall?.copyWith(
            fontWeight: FontWeight.w700,
            color: MedicalTokens.textMain,
          ),
          headlineSmall: baseText.headlineSmall?.copyWith(
            fontWeight: FontWeight.w700,
            color: MedicalTokens.textMain,
          ),
          titleLarge: baseText.titleLarge?.copyWith(fontWeight: FontWeight.w700),
          titleMedium: baseText.titleMedium?.copyWith(fontWeight: FontWeight.w700),
          titleSmall: baseText.titleSmall?.copyWith(fontWeight: FontWeight.w600),
          bodyMedium: baseText.bodyMedium?.copyWith(color: MedicalTokens.textMain),
          bodySmall: baseText.bodySmall?.copyWith(color: MedicalTokens.textSubtle),
        ),
        appBarTheme: AppBarTheme(
          centerTitle: false,
          elevation: 0,
          scrolledUnderElevation: 0,
          backgroundColor: MedicalTokens.background,
          foregroundColor: colorScheme.onSurface,
          titleTextStyle: baseText.titleLarge?.copyWith(
            fontWeight: FontWeight.w800,
            color: MedicalTokens.textMain,
          ),
        ),
        cardTheme: CardThemeData(
          elevation: 0,
          color: MedicalTokens.surface,
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(MedicalTokens.radiusLg),
            side: const BorderSide(color: MedicalTokens.border),
          ),
        ),
        elevatedButtonTheme: ElevatedButtonThemeData(
          style: ElevatedButton.styleFrom(
            minimumSize: const Size.fromHeight(50),
            textStyle: baseText.labelLarge?.copyWith(fontWeight: FontWeight.w700),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
            ),
          ),
        ),
        filledButtonTheme: FilledButtonThemeData(
          style: FilledButton.styleFrom(
            backgroundColor: MedicalTokens.primary,
            foregroundColor: Colors.white,
            minimumSize: const Size.fromHeight(50),
            textStyle: baseText.labelLarge?.copyWith(fontWeight: FontWeight.w700),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
            ),
          ),
        ),
        outlinedButtonTheme: OutlinedButtonThemeData(
          style: OutlinedButton.styleFrom(
            foregroundColor: MedicalTokens.textMain,
            side: const BorderSide(color: MedicalTokens.border),
            minimumSize: const Size.fromHeight(50),
            textStyle: baseText.labelLarge?.copyWith(fontWeight: FontWeight.w700),
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(MedicalTokens.radiusMd),
            ),
          ),
        ),
      ),
      initialRoute: '/upload',
      routes: {
        '/upload': (_) => const UploadScreen(),
        '/history': (_) => const HistoryScreen(),
        '/result': (context) {
          final args = ModalRoute.of(context)!.settings.arguments;
          if (args is ResultScreenArgs) {
            return ResultScreen(args: args);
          }
          if (args is Uint8List) {
            return ResultScreen(originalImageBytes: args);
          }
          return const ResultScreen();
        },
      },
    ),
  );
}
