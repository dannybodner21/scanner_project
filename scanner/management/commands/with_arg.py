def predict_short_vertex_new(request):
    try:
        access_token = get_vertex_access_token()
    except Exception as e:
        return JsonResponse({"status": "error", "message": f"Auth failed: {e}"}, status=500)

    # get recent Metrics
    cutoff = now() - timedelta(minutes=5)
    metrics = RickisMetrics.objects.filter(timestamp__gte=cutoff)

    instances = []
    symbols = []

    for metric in metrics:
        try:
            # gather necessary data
            instance = {
                "price": float(metric.price),
                "volume": float(metric.volume),
                "change_5m": float(metric.change_5m),
                "change_1h": float(metric.change_1h),
                "change_24h": float(metric.change_24h),
                "high_24h": float(metric.high_24h),
                "low_24h": float(metric.low_24h),
                "open": float(metric.open),
                "close": float(metric.close),
                "avg_volume_1h": float(metric.avg_volume_1h),
                "relative_volume": float(metric.relative_volume),
                "sma_5": float(metric.sma_5),
                "sma_20": float(metric.sma_20),
                "macd": float(metric.macd),
                "macd_signal": float(metric.macd_signal),
                "rsi": float(metric.rsi),
                "stochastic_k": float(metric.stochastic_k),
                "stochastic_d": float(metric.stochastic_d),
                "support_level": float(metric.support_level),
                "resistance_level": float(metric.resistance_level),
                "stddev_1h": float(metric.stddev_1h),
                "price_slope_1h": float(metric.price_slope_1h),
                "atr_1h": float(metric.atr_1h),
                "obv": float(metric.obv),
                "change_since_high": float(metric.change_since_high),
                "change_since_low": float(metric.change_since_low),
                "fib_distance_0_236": float(metric.fib_distance_0_236),
                "fib_distance_0_382": float(metric.fib_distance_0_382),
                "fib_distance_0_5": float(metric.fib_distance_0_5),
                "fib_distance_0_618": float(metric.fib_distance_0_618),
                "fib_distance_0_786": float(metric.fib_distance_0_786),
            }
            instances.append(instance)
            symbols.append(metric.coin.symbol)


        except Exception:
            continue

    if not instances:
        return JsonResponse({"status": "error", "message": "No valid instances"}, status=500)

    url = (
        f"https://{SHORT_REGION}-aiplatform.googleapis.com/v1/"
        f"projects/{SHORT_PROJECT_ID}/locations/{SHORT_REGION}/endpoints/{SHORT_ENDPOINT_ID}:predict"
    )
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
    }

    try:
        # send to GCP
        response = requests.post(url, headers=headers, json={"instances": instances})
        response.raise_for_status()
        predictions = response.json()
        messages = []

        # print out confidence per coin
        for symbol, result in zip(symbols, predictions["predictions"]):
            true_index = result["classes"].index("true")
            confidence = result["scores"][true_index]
            print(f"SHORT: {symbol} — Confidence: {confidence:.4f}")

            # send message through Telegram if confidence is greater than 0.6
            if confidence > 0.6:
                messages.append(f"SHORT | {symbol} — Confidence: {confidence:.4f}")

        # telegram alert
        if len(messages) > 0:
            send_text(messages)

        return JsonResponse({"status": "success", "predictions": predictions})

    except Exception as e:
        return JsonResponse({
            "status": "error",
            "message": str(e),
            "response": getattr(response, "text", "No response")
        }, status=500)
