# ここに公式サーバーのバイナリを置いてください

第37回プロコン公式サイトで配布されている「簡易サーバー.zip」を展開し、
中の `procon-server-*` をこのフォルダに置いてください。

  https://www.procon.gr.jp/news/2026/08/20/117

置くと、メニューの「7. 公式サーバーと照合する」と
`tests/test_official_server.py` が動くようになります。
置いていない場合、そのテストは自動でスキップされます。

ライセンスの都合により、バイナリ本体はこのプロジェクトに同梱していません。

| OS | ファイル名 |
|---|---|
| Linux (amd64) | procon-server-linux-amd64 |
| macOS (Apple Silicon) | procon-server-darwin-arm64 |
| macOS (Intel) | procon-server-darwin-amd64 |
| Windows (amd64) | procon-server-windows-amd64.exe |

Linux と macOS では実行権限が要ります。

    chmod +x server/procon-server-*
