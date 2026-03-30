import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'screens/metrics_screen.dart';
import 'screens/result_screen.dart';
import 'screens/upload_screen.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      initialRoute: '/upload',
      routes: {
        '/upload': (_) => const UploadScreen(),
        '/result': (context) {
          final args = ModalRoute.of(context)!.settings.arguments;
          final bytes = args is Uint8List ? args : null;
          return ResultScreen(originalImageBytes: bytes);
        },
        '/metrics': (_) => const MetricsScreen(),
      },
    ),
  );
}
