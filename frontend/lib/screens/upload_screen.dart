import 'dart:typed_data';

import 'package:file_picker/file_picker.dart';
import 'package:flutter/material.dart';

class UploadScreen extends StatefulWidget {
  const UploadScreen({super.key});

  @override
  State<UploadScreen> createState() => _UploadScreenState();
}

class _UploadScreenState extends State<UploadScreen> {
  String? fileName;
  Uint8List? fileBytes;

  Future<void> pickFile() async {
    final result = await FilePicker.platform.pickFiles(
      type: FileType.image,
      allowMultiple: false,
      withData: true,
    );
    if (!mounted) return;
    if (result == null || result.files.isEmpty) return;

    final f = result.files.single;
    final bytes = f.bytes;
    if (bytes == null) return;

    setState(() {
      fileBytes = bytes;
      fileName = f.name;
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Upload Retinal Image')),
      body: Center(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            fileName != null
                ? Text('Selected: $fileName')
                : const Text('No image selected'),
            const SizedBox(height: 20),
            if (fileBytes != null)
              Image.memory(fileBytes!, width: 250, height: 250, fit: BoxFit.contain),

            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: pickFile,
              child: const Text('Select Image'),
            ),

            const SizedBox(height: 20),
            ElevatedButton(
              onPressed: fileBytes == null
                  ? null
                  : () {
                      Navigator.pushNamed(context, '/result', arguments: fileBytes);
                    },
              child: const Text('Next'),
            ),
          ],
        ),
      ),
    );
  }
}
