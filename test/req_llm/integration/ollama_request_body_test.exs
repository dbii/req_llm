defmodule ReqLLM.Integration.OllamaRequestBodyTest do
  @moduledoc """
  Live integration test that round-trips a real request against a local Ollama.

  Regression coverage for the empty-request-body bug: the built-in `:ollama`
  provider used to send `Content-Length: 0` (the encoded JSON body never reached
  the outgoing request), so Ollama rejected every call with
  `400 {"error":{"message":"EOF","type":"invalid_request_error"}}`.

  This test is gated on a reachable Ollama instance and skipped otherwise, so it
  never breaks CI where no Ollama is running.

  Run with:

      mix test test/req_llm/integration/ollama_request_body_test.exs --include integration
  """

  use ExUnit.Case, async: false

  @moduletag :integration
  @moduletag timeout: 120_000

  @model "ollama:llama3.1"
  @base_url "http://localhost:11434"

  @ollama_up? match?(
                {:ok, %{status: 200}},
                Req.get(@base_url <> "/api/tags", retry: false, receive_timeout: 1_000)
              )

  if not @ollama_up? do
    @moduletag skip: "Local Ollama not reachable at #{@base_url}"
  end

  test "generate_text round-trips against live Ollama with a non-empty body" do
    case ReqLLM.generate_text(@model, "Reply with the single word: pong") do
      {:ok, response} ->
        assert %ReqLLM.Response{} = response
        text = ReqLLM.Response.text(response)
        assert is_binary(text)
        assert String.trim(text) != ""

      {:error, error} ->
        flunk("""
        Ollama rejected the request — likely an empty request body regression:
        #{inspect(error, pretty: true)}
        """)
    end
  end

  test "stream_text round-trips against live Ollama with a non-empty body" do
    case ReqLLM.stream_text(@model, "Reply with the single word: pong") do
      {:ok, stream_response} ->
        text = stream_response |> ReqLLM.StreamResponse.tokens() |> Enum.to_list() |> Enum.join()
        assert String.trim(text) != ""

      {:error, error} ->
        flunk("""
        Ollama rejected the streaming request — likely an empty request body regression
        in encode_stream_body/3 (missing Req.Steps.encode_body/1 materialization step):
        #{inspect(error, pretty: true)}
        """)
    end
  end

  # A reasoning model (qwen3.x) emits <think> traces by default, which breaks
  # structured output and blows up latency. `reasoning_effort: "none"` (a core
  # ReqLLM option this provider now forwards to the Ollama body) disables that.
  # Skipped unless a qwen3 model is pulled locally.
  test "reasoning_effort: \"none\" yields clean structured output from a reasoning model" do
    reasoning_model =
      case Req.get(@base_url <> "/api/tags", retry: false, receive_timeout: 1_000) do
        {:ok, %{status: 200, body: %{"models" => models}}} ->
          models |> Enum.map(& &1["name"]) |> Enum.find(&String.starts_with?(&1, "qwen3"))

        _ ->
          nil
      end

    if reasoning_model == nil do
      # No reasoning model available — nothing to exercise.
      assert true
    else
      schema = [city: [type: :string, required: true]]

      case ReqLLM.generate_object(
             "ollama:" <> reasoning_model,
             "Marcus lives in Portland. Return the city.",
             schema,
             reasoning_effort: :none,
             temperature: 0.0
           ) do
        {:ok, response} ->
          object = ReqLLM.Response.object(response)
          assert is_map(object)
          assert object["city"] =~ "Portland"

        {:error, error} ->
          flunk("""
          generate_object with reasoning_effort: "none" failed on #{reasoning_model}.
          Either the option is not being forwarded to the Ollama body, or thinking
          traces broke structured-output parsing:
          #{inspect(error, pretty: true)}
          """)
      end
    end
  end
end
