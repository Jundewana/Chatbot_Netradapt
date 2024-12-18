import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(QuizBotApp());
}

class QuizBotApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'QuizBot - Netradapt',
      theme: ThemeData(
        primarySwatch: Colors.green,
      ),
      home: ChatPage(),
    );
  }
}

class ChatPage extends StatefulWidget {
  @override
  _ChatPageState createState() => _ChatPageState();
}

class _ChatPageState extends State<ChatPage> {
  final TextEditingController _controller = TextEditingController();
  final List<Map<String, String>> _messages = [];
  bool _isLoading = false;
  String _apiMessage = '';

  Future<void> _sendMessage(String message) async {
    setState(() {
      _isLoading = true;
      _messages.add({'sender': 'user', 'text': message});
    });

    try {
      final response = await http.post(
        Uri.parse('http://192.168.1.38:3000/chat'),
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'userInput': message}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _messages.add({'sender': 'bot', 'text': data['response']});
        });
      } else {
        setState(() {
          _messages.add(
              {'sender': 'bot', 'text': 'Error: Server returned an error!'});
        });
      }
    } catch (error) {
      setState(() {
        _messages.add({
          'sender': 'bot',
          'text': 'Error: Failed to connect to the server!'
        });
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  Future<void> _fetchApiMessage() async {
    try {
      final response =
          await http.get(Uri.parse('http://192.168.1.38:3000/api/public'));

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        setState(() {
          _apiMessage =
              data['signature']; // Ubah kunci 'message' menjadi 'signature'
        });
      } else {
        setState(() {
          _apiMessage = 'Error: Server returned an error!';
        });
      }
    } catch (error) {
      setState(() {
        _apiMessage = 'Error: Failed to connect to the server!';
      });
    }
  }

  @override
  void initState() {
    super.initState();
    _fetchApiMessage();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('QuizBot - Netradapt'),
      ),
      body: Column(
        children: [
          Text('API Message: $_apiMessage'),
          Expanded(
            child: ListView.builder(
              itemCount: _messages.length,
              itemBuilder: (context, index) {
                final message = _messages[index];
                return ListTile(
                  title: Align(
                    alignment: message['sender'] == 'user'
                        ? Alignment.centerRight
                        : Alignment.centerLeft,
                    child: Container(
                      padding: EdgeInsets.all(10),
                      decoration: BoxDecoration(
                        color: message['sender'] == 'user'
                            ? Colors.blue
                            : Colors.green,
                        borderRadius: BorderRadius.circular(10),
                      ),
                      child: Text(
                        message['text']!,
                        style: TextStyle(color: Colors.white),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),
          if (_isLoading) CircularProgressIndicator(),
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _controller,
                    decoration: InputDecoration(
                      hintText: 'Tanyakan sesuatu...',
                      border: OutlineInputBorder(),
                    ),
                  ),
                ),
                SizedBox(width: 8),
                ElevatedButton(
                  onPressed: () {
                    final message = _controller.text.trim();
                    if (message.isNotEmpty) {
                      _controller.clear();
                      _sendMessage(message);
                    }
                  },
                  child: Text('Kirim'),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}
