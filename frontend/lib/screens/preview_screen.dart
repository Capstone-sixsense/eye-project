import 'dart:typed_data';

import 'package:flutter/material.dart';

class PreviewScreen extends StatelessWidget {
  const PreviewScreen({super.key, this.imageBytes});

  final Uint8List? imageBytes;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Preview')),
      body: Center(
        child: imageBytes != null
            ? Image.memory(imageBytes!, fit: BoxFit.contain)
            : const Text('No image'),
      ),
    );
  }
}
