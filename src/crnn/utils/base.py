def get_duration_str(start_time: float, end_time: float) -> str:
    duration = end_time - start_time
    days, remainder = divmod(duration, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(days):02}:{int(hours):02}:{int(minutes):02}:{int(seconds):02}"
