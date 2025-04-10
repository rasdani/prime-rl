from typing import Any
import aiohttp
from zeroband.logger import get_logger
import asyncio
from zeroband.training import envs


async def _get_external_ip(max_retries=3, retry_delay=5):
    async with aiohttp.ClientSession() as session:
        for attempt in range(max_retries):
            try:
                async with session.get("https://api.ipify.org", timeout=10) as response:
                    response.raise_for_status()
                    return await response.text()
            except aiohttp.ClientError:
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
    return None


class HttpMonitor:
    """
    Logs the status of nodes, and training progress to an API
    """

    def __init__(self, log_flush_interval: int = 10):
        self.data = []
        self.log_flush_interval = log_flush_interval
        self.base_url = envs.PRIME_API_BASE_URL
        self.auth_token = envs.PRIME_DASHBOARD_AUTH_TOKEN

        self._logger = get_logger()

        self.run_id = "b396001defcf4a3d9181b893de80573b"
        if self.run_id is None:
            raise ValueError("run_id must be set for HttpMonitor")

        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def __del__(self):
        self.loop.close()

    def _remove_duplicates(self):
        seen = set()
        unique_logs = []
        for log in self.data:
            log_tuple = tuple(sorted(log.items()))
            if log_tuple not in seen:
                unique_logs.append(log)
                seen.add(log_tuple)
        self.data = unique_logs

    def log(self, data: dict[str, Any]):
        # Lowercase the keys in the data dictionary
        lowercased_data = {k.lower(): v for k, v in data.items()}
        self.data.append(lowercased_data)

        self._handle_send_batch()

    def _handle_send_batch(self, flush: bool = False):
        if len(self.data) >= self.log_flush_interval or flush:
            self.loop.run_until_complete(self._send_batch())

    async def _send_batch(self):
        self._remove_duplicates()

        batch = self.data[: self.log_flush_interval]
        filtered_batch = [{k: d[k] for k in ("step", "seq_lens", "sample_reward")} for d in batch]
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.auth_token}"}
        payload = {"metrics": filtered_batch, "operation_type": "append"}
        api = f"{self.base_url}/pools/{self.run_id}/metrics"

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(api, json=payload, headers=headers) as response:
                    if response is not None:
                        response.raise_for_status()
        except Exception as e:
            self._logger.error(f"Error sending batch to server: {str(e)}")
            pass

        self.data = self.data[self.log_flush_interval :]
        return True

    async def _finish(self):
        # Send any remaining logs
        while self.data:
            await self._send_batch()

        return True

    def finish(self):
        self.loop.run_until_complete(self._finish())
