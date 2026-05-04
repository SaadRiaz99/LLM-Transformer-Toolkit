class Agent:
    def reply(self, message):
        message = message.lower()

        if "hello" in message:
            return "Hi "
        elif "bye" in message or "by by" in message:
            return "Allah Hafiz "
        else:
            return "I don't understand "


bot = Agent()

while True:
    message = input("Enter (hello / bye / exit): ")

    if message.lower() == "exit":
        print("Bot: Goodbye! ")
        break

    response = bot.reply(message)
    print("Bot:", response)