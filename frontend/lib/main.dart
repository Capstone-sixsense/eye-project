import 'dart:typed_data';

import 'package:flutter/material.dart';

import 'models/result_screen_args.dart';
import 'screens/history_screen.dart';
import 'screens/result_screen.dart';
import 'screens/upload_screen.dart';

void main() {
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
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
