import discord
from discord.ext import commands
from discord import FFmpegPCMAudio

from authtokens import *

class dimir(commands.Bot):

    def __init__(self, command_prefix, self_bot):
        intents = discord.Intents.all()
        intents.members = True
        commands.Bot.__init__(self, command_prefix = command_prefix, self_bot = self_bot, intents = intents)
        self.message1 = '[INFO]: Bot now online'
        self.message2 = 'Bot still online'
        self.add_commands()

    async def on_ready(self):
        print(self.message1)

    def add_commands(self):
        @self.command(name = 'status', pass_context = True)
        async def status(ctx):
            print(ctx)
            await ctx.channel.send(f'{self.message2}, {ctx.author.name}')

        @self.command(name = 'hello', pass_context = True)
        async def hello(ctx):
            await ctx.channel.send(f'Hello {ctx.author.name}, I am Doctor RP and I am a bot.')

        @self.command(name = 'join', pass_context = True)
        async def join(ctx):
            if ctx.author.voice:
                channel = ctx.message.author.voice.channel
                voice = await channel.connect()
            else:
                await ctx.send('[ERROR]: you\'re not in a voice channel.')

        @self.command(name = 'leave', pass_context = True)
        async def leave(ctx):
            if ctx.voice_client:
                await ctx.guild.voice_client.disconnect()
                await ctx.send('[INFO]: leaving voice channel')
            else:
                await ctx.send('[ERROR]: I\'m not in a voice channel.')

        @self.command(name = 'pause', pass_context = True)
        async def pause(ctx):
            voice = discord.utils.get(self.voice_clients, guild = ctx.guild)
            if voice.is_playing():
                voice.pause()
            else:
                pass

        @self.command(name = 'resume', pass_context = True)
        async def resume(ctx):
            voice = discord.utils.get(self.voice_clients, guild = ctx.guild)
            if voice.is_paused():
                voice.resume()
            else:
                pass

        @self.command(name = 'stop', pass_context = True)
        async def stop(ctx):
            voice = discord.utils.get(self.voice_clients, guild = ctx.guild)
            voice.stop()


        @self.command(name = 'play', pass_context = True)
        async def play(ctx, arg):
            self.queue = {}
            if not ctx.voice_client and ctx.author.voice:
                channel = ctx.message.author.voice.channel
                voice = await channel.connect()
                source = FFmpegPCMAudio(arg)
                player = voice.play(source, after = lambda x = None: self.check_queue(ctx, ctx.message.guild.id))
            else:
                voice = ctx.guild.voice_client
                source = FFmpegPCMAudio(arg)
                player = voice.play(source, after = lambda x = None: self.check_queue(ctx, ctx.message.guild.id))

        @self.command(name = 'queue', pass_context = True)
        async def queue(ctx, arg):
            voice = ctx.guild.voice_client
            source = FFmpegPCMAudio(arg)

            guild_id = ctx.message.guild.id

            if guild_id in self.queue:
                self.queue[giuld_id].append(source)
            else:
                self.queue[guild_id] = [source]

            await ctx.send(f'Added {source} to queue')
            print(self.queue)

    def check_queue(self, ctx, id):
        if self.queue[id] != []:
            voice = ctx.guild.voice_client
            source = self.queue[id].pop(0)
            player = voice.play(source)






def main():
    prefix = ['!', '?', 'System: ']
    dimir(prefix, False).run(token_bot)


if __name__ == "__main__":
    main()