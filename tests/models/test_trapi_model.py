from unittest.mock import Mock, patch

from minisweagent.models.trapi_model import TrapiModel


def test_trapi_model_init():
    """Test TrapiModel initialization with default values."""
    model = TrapiModel()
    assert model.trapi_config.model_name == "gpt-4o_2024-11-20"
    assert model.trapi_config.instance == "gcr/preview"
    assert model.trapi_config.api_version == "2024-12-01-preview"
    assert model.cost == 0.0
    assert model.n_calls == 0


def test_trapi_model_init_with_custom_config():
    """Test TrapiModel initialization with custom configuration."""
    custom_model = "gpt-4o-custom"
    custom_instance = "custom/instance"
    custom_api_version = "2024-10-01-preview"
    
    model = TrapiModel(
        model_name=custom_model,
        instance=custom_instance,
        api_version=custom_api_version,
        model_kwargs={"temperature": 0.5}
    )
    
    assert model.trapi_config.model_name == custom_model
    assert model.trapi_config.instance == custom_instance
    assert model.trapi_config.api_version == custom_api_version
    assert model.trapi_config.model_kwargs == {"temperature": 0.5}


@patch('minisweagent.models.trapi_model.AzureOpenAI')
@patch('minisweagent.models.trapi_model.get_bearer_token_provider')
def test_trapi_model_query(mock_token_provider, mock_azure_openai):
    """Test TrapiModel query method."""
    # Mock the response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response"
    
    # Mock the client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client
    
    # Mock token provider
    mock_token_provider.return_value = Mock()
    
    model = TrapiModel(model_name="test-model")
    
    messages = [{"role": "user", "content": "Test message"}]
    result = model.query(messages)
    
    assert result["content"] == "Test response"
    assert model.n_calls == 1
    
    # Verify client was called correctly
    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=messages
    )


@patch('minisweagent.models.trapi_model.AzureOpenAI')
@patch('minisweagent.models.trapi_model.get_bearer_token_provider')
def test_trapi_model_query_with_kwargs(mock_token_provider, mock_azure_openai):
    """Test TrapiModel query method with additional kwargs."""
    # Mock the response
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message.content = "Test response with kwargs"
    
    # Mock the client
    mock_client = Mock()
    mock_client.chat.completions.create.return_value = mock_response
    mock_azure_openai.return_value = mock_client
    
    # Mock token provider
    mock_token_provider.return_value = Mock()
    
    model = TrapiModel(
        model_name="test-model",
        model_kwargs={"temperature": 0.7}
    )
    
    messages = [{"role": "user", "content": "Test message"}]
    result = model.query(messages, max_tokens=100)
    
    assert result["content"] == "Test response with kwargs"
    assert model.n_calls == 1
    
    # Verify client was called with merged kwargs
    mock_client.chat.completions.create.assert_called_once_with(
        model="test-model",
        messages=messages,
        temperature=0.7,
        max_tokens=100
    )
