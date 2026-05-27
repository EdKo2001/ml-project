from src.models import get_model
m = get_model('mlp')
print(type(m), getattr(m, 'random_state', None))
