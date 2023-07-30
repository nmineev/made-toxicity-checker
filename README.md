# Toxicity Checker Bot
### Telegram Bot for toxic messages detection.
Can be added to a telegram group and will inform group members when some message 
toxicity exceeds acceptable, by administrators, toxicity threshold, so moderation 
of such group significantly simplifies.  

### Functionality:
1. Automatically checks every message in a group (or a chat) for acceptable toxicity 
percent exceeding and, when it happens, recognizes this message as toxic and sends 
informing message.
2. Administrators of a group can vary acceptable toxicity percent by command 
`/set_acceptable_toxicity_percent <int between 0 and 100>`.  
By default acceptable 
percent is 70.
3. Other users can view acceptable toxicity percent in a group through 
`/acceptable_toxicity_percent` command.
4. Show toxicity percent of arbitrary text with command `/toxicity <text>`
5. Show toxicity percent of any replayed message with command `/toxicity`

### Deployment
1. Create your telegram bot (as described [here](https://core.telegram.org/bots/features#creating-a-new-bot))
and write your bot token into `.env` file:  
`TOKEN=<your_tgbottoken>`
2. Run in terminal: `sudo docker compose up --build`

### Demonstration
![demo](demka_final.gif)
