import discord
from discord.ext import commands

class DiscordBotRunner:
    def __init__(self, token):
        self.token = token
        self.bot = commands.Bot(command_prefix='!')

    async def send_message(self, channel_id, message):
        channel = self.bot.get_channel(channel_id)
        if channel:
            await channel.send(message)

    async def receive_message(self, channel_id):
        channel = self.bot.get_channel(channel_id)
        if channel:
            async for message in channel.history(limit=1):
                return message.content
        return None

    def run(self):
        self.bot.run(self.token)