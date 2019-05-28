import logging
import socket
import time

from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

try:
    from imjoy import api
except ImportError:

    class ImjoyApi:
        def log(self, msg):
            logger.info(msg)

        def alert(self, msg):
            logger.warning(msg)

        def showDialog(self, *args, **kwargs):
            print(args, kwargs)

    api = ImjoyApi()

from tiktorch.launcher import LocalServerLauncher, RemoteSSHServerLauncher, SSHCred
from tiktorch.rpc_interface import INeuralNetworkAPI, IFlightControl
from tiktorch.rpc import Client, TCPConnConf


@dataclass
class Config:
    address = "localhost"
    port1 = 5556
    port2 = 5557
    username = "username"
    key_path = "ssh key"
    # address = "gpu6.cluster.embl.de"
    # port1 = 5556
    # port2 = 5557
    # username = "beuttenm"
    # key_path = "/Users/fbeut/.ssh/id_rsa"


@dataclass()
class Ctx:
    config = Config()


class ImJoyPlugin(INeuralNetworkAPI):
    def setup(self) -> None:
        api.log("initialized")

    async def _choose_devices(self, data) -> None:
        await api.alert(str(data))

    async def run(self, ctx) -> None:
        ctx = Ctx()
        address = socket.gethostbyname(ctx.config.address)
        port1 = str(ctx.config.port1)
        port2 = str(ctx.config.port2)

        conn_conf = TCPConnConf(address, port1, port2)

        if address == "127.0.0.1":
            self.launcher = LocalServerLauncher(conn_conf, dummy=True)
        else:
            self.launcher = RemoteSSHServerLauncher(
                conn_conf, cred=SSHCred(ctx.config.username, key_path=ctx.config.key_path), dummy=True
            )

        api.log(f"start server at {address}:{port1};{port2}")
        self.launcher.start()
        api.log("server started")
        c = Client(IFlightControl(), conn_conf)
        api.log(f"ping {c.ping()}")
        api.alert(f"server running at {address}:{port1};{port2}: {self.launcher.is_server_running()}")

        try:
            tikTorchClient = Client(INeuralNetworkAPI(), conn_conf)
            available_devices = tikTorchClient.get_available_devices()
        except Exception as e:
            self.launcher.stop()
            logger.exception(e)
            return

        api.log(f"available devices: {available_devices}")
        device_switch_template = {
            "type": "switch",
            "label": "Device",
            "model": "status",
            "multi": True,
            "readonly": False,
            "featured": False,
            "disabled": False,
            "default": False,
            "textOn": "Selected",
            "textOff": "Not Selected",
        }

        choose_devices_schema = {"fields": [device_switch_template]}
        answer = await api.showDialog(
            {
                "name": "Select from available devices",
                "type": "SchemaIO",
                "w": 40,
                "h": 15,
                "data": {
                    "title": f"Select devices for TikTorch server at {address}",
                    "schema": choose_devices_schema,
                    "model": {},
                    "callback": self._choose_devices,
                    "show": True,
                    "formOptions": {"validateAfterLoad": True, "validateAfterChanged": True},
                    "id": 0,
                },
            }
        )

        time.sleep(10)
        await self.exit()

    async def exit(self):
        self.launcher.stop()
        api.log("server stopped")

    def resume_training(self) -> None:
        pass


if __name__ == "__main__":
    import asyncio

    logging.basicConfig(level=logging.DEBUG)

    loop = asyncio.get_event_loop()
    # ctx = Ctx()
    plugin = ImJoyPlugin()
    plugin.setup()
    loop.run_until_complete(plugin.run(ctx))
    loop.run_until_complete(plugin.exit())
