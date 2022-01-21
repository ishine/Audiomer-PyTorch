from datamodules.SpeechCommands12 import SpeechCommands12

ds = SpeechCommands12(
    "testing", "./")
print(ds[0])
