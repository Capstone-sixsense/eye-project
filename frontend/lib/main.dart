import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'screens/metrics_screen.dart';
import 'screens/preview_screen.dart';
import 'screens/result_screen.dart';
import 'screens/upload_screen.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/upload',
      routes: {
        '/upload': (_) => const UploadScreen(),
        '/preview': (context) {
          final args = ModalRoute.of(context)!.settings.arguments;
          final bytes = args is Uint8List ? args : null;
          return PreviewScreen(imageBytes: bytes);
        },
        '/result': (_) => const ResultScreen(),
        '/metrics': (_) => const MetricsScreen(),
      },
    ),
  );
}
