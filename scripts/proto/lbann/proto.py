"""Generate prototexts for LBANN models."""

# Check for Python 3
import sys
if sys.version_info[0] != 3:
    raise ImportError('Python 3 is required')

# Import modules
import google.protobuf.text_format
from collections.abc import Iterable

# Import lbann_pb2 module generated by protobuf
# Note: This should be built automatically during the LBANN build
# process. If it's not in the default Python search path, try to find
# it with some heuristics.
try:
    import lbann_pb2
except ImportError:
    # Not found, try to find and add the build directory for this system.
    import socket, re, os, os.path
    _system_name = re.sub(r'\d+', '', socket.gethostname())
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _lbann_dir = os.path.dirname(os.path.dirname(os.path.dirname(_script_dir)))
    # For now, hardcode GCC, Release/Debug, and .llnl.gov.
    # TODO: Relax this.
    _release_dir = os.path.join(_lbann_dir, 'build',
                                'gnu.Release.' + _system_name + '.llnl.gov')
    _debug_dir = os.path.join(_lbann_dir, 'build',
                              'gnu.Debug.' + _system_name + '.llnl.gov')
    if os.path.isdir(_release_dir):
        sys.path.append(os.path.join(_release_dir,
                                     'install', 'share', 'python'))
        import lbann_pb2
    elif os.path.isdir(_debug_dir):
        sys.path.append(os.path.join(_debug_dir,
                                     'install', 'share', 'python'))
        import lbann_pb2
    else:
        raise  # Give up.

def _add_to_module_namespace(stuff):
    """Add stuff to the module namespace.

    stuff is a dict, keys will be the name.

    """
    g = globals()
    for k, v in stuff.items():
        g[k] = v

def _make_iterable(obj):
    """Convert to an iterable object.

    Simply returns `obj` if it is alredy iterable. Otherwise returns a
    1-tuple containing `obj`.

    """
    if isinstance(obj, Iterable):
        return obj
    else:
        return (obj,)

# Map protobuf label enums to a string name.
_proto_label_to_str = {
    google.protobuf.descriptor.FieldDescriptor.LABEL_OPTIONAL: 'optional',
    google.protobuf.descriptor.FieldDescriptor.LABEL_REPEATED: 'repeated'
}
# Map protobuf type enums to a strong name.
_proto_type_to_str = {
    google.protobuf.descriptor.FieldDescriptor.TYPE_BOOL: 'bool',
    google.protobuf.descriptor.FieldDescriptor.TYPE_BYTES: 'bytes',
    google.protobuf.descriptor.FieldDescriptor.TYPE_DOUBLE: 'double',
    google.protobuf.descriptor.FieldDescriptor.TYPE_ENUM: 'enum',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED32: 'fixed32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FIXED64: 'fixed64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_FLOAT: 'float',
    google.protobuf.descriptor.FieldDescriptor.TYPE_GROUP: 'group',
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT32: 'int32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_INT64: 'int64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_MESSAGE: 'message',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED32: 'sfixed32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SFIXED64: 'sfixed64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SINT32: 'sint32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_SINT64: 'sint64',
    google.protobuf.descriptor.FieldDescriptor.TYPE_STRING: 'string',
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT32: 'uint32',
    google.protobuf.descriptor.FieldDescriptor.TYPE_UINT64: 'uint64'
}

def _generate_class(base_class, type_name, message_field_name,
                    base_kwargs=[], type_has_parent=True):
    """Generate a new class from Protobuf.

    base_class is the class the generated class will inherit from.
    base_class should have __init__ and export_proto methods. export_proto is
    only used if type_has_parent is True.
    type_name is the name of the Protobuf type to generate the class from.
    message_field_name is the name of the field in the Protobuf message.
    base_kwargs is a list of (arg, default value) kwargs that will be passed to
    the base_class's __init__ method instead of being treated as field names.
    type_has_parent indicates whether this message is nested within a parent
    message.

    Returns a new class type.

    """
    # Extract the names of all fields in the type.
    message_type = getattr(lbann_pb2, type_name)
    field_names = list(message_type.DESCRIPTOR.fields_by_name.keys())

    # Define the constructor.
    def __init__(self, *args, **kwargs):
        # Extract arguments to pass to the base class __init__, accounting for
        # regular args.
        init_kwargs = dict(base_kwargs[len(args):])
        for arg_name, _ in base_kwargs:
            if arg_name in field_names:
                raise RuntimeError('Keyword arg {0} matches existing field for {1}. This is a bug!'.format(
                    arg_name, type_name))
            if arg_name in kwargs:
                init_kwargs[arg_name] = kwargs[arg_name]
                del kwargs[arg_name]
        base_class.__init__(self, *args, **init_kwargs)
        # Check and set up fields.
        for field in kwargs:
            if field not in field_names:
                raise ValueError('Unknown argument {0}'.format(field))
        for field_name in field_names:
            # Ensure we don't accidentally clobber an existing variable.
            try:
                getattr(self, field_name)
                raise RuntimeError('Field {0} conflicts with already existing field for {1}. This is a bug!'.format(
                    field_name, type_name))
            except: pass
            # Set field values.
            if field_name in kwargs:
                setattr(self, field_name, kwargs[field_name])
            else:
                setattr(self, field_name, None)
    # Define the method to export a protobuf message.
    def export_proto(self):
        if type_has_parent:
            proto = base_class.export_proto(self)
            message = getattr(proto, message_field_name)
            message.SetInParent()  # Create empty message.
        else:
            proto = message_type()
            message = proto
        for field_name in field_names:
            v = getattr(self, field_name)
            if v is not None:
                if type(v) is list:  # Repeated field.
                    getattr(message, field_name).extend(v)
                else:  # Singular field.
                    setattr(message, field_name, v)
        return proto
    # Define the method to return the names of all fields.
    def get_field_names(self):
        return field_names
    # Define a simple docstring consisting of the available fields.
    if field_names:
        doc = 'Fields:\n'
        for field_name in field_names:
            doc += '{0} ({1} {2})\n'.format(
                field_name,
                _proto_label_to_str.get(
                    message_type.DESCRIPTOR.fields_by_name[field_name].label,
                    'unknown'),
                _proto_type_to_str.get(
                    message_type.DESCRIPTOR.fields_by_name[field_name].type,
                    'unknown'))
    else:
        doc = 'Fields: n/a\n'
    # Create the sub-class.
    return type(type_name, (base_class,),
                {'__init__': __init__, 'export_proto': export_proto,
                 '__doc__': doc,
                 'get_field_names': get_field_names})

def _generate_classes_from_message(base_class, message, skip_fields=None,
                                   base_kwargs=[], type_has_parent=True):
    """Generate new classes based on fields in message.

    base_class is the class generated classes will inherit from.
    message is a Protobuf message type (e.g. lbann_pb2.Layer).
    skip_fields is a set of field names to not generate classes for.
    base_kwargs and type_has_parent are passed to _generate_class.

    Classes are automatically added to the namespace.

    """
    skip_fields = skip_fields or set()
    generated_classes = {}
    for field in message.DESCRIPTOR.fields:
        if field.name not in skip_fields:
            type_name = field.message_type.name
            generated_classes[type_name] = _generate_class(
                base_class, type_name, field.name,
                base_kwargs=base_kwargs, type_has_parent=type_has_parent)
    _add_to_module_namespace(generated_classes)

# ==============================================
# Layers
# ==============================================

class Layer:
    """Base class for layers."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, parents, children, weights,
                 name, data_layout, hint_layer):
        Layer.global_count += 1
        self.parents = []
        self.children = []
        self.weights = []
        self.name = name if name else 'layer{0}'.format(Layer.global_count)
        self.data_layout = data_layout
        self.hint_layer = hint_layer

        # Initialize parents, children, and weights
        for l in _make_iterable(parents):
            self.add_parent(l)
        for l in _make_iterable(children):
            self.add_child(child)
        for w in _make_iterable(weights):
            self.add_weights(w)

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Layer()
        proto.parents = ' '.join([l.name for l in self.parents])
        proto.children = ' '.join([l.name for l in self.children])
        proto.weights = ' '.join([w.name for w in self.weights])
        proto.name = self.name
        proto.data_layout = self.data_layout
        proto.hint_layer = self.hint_layer.name if self.hint_layer else ''
        return proto

    def add_parent(self, parent):
        """This layer will receive an input tensor from `parent`."""
        for p in _make_iterable(parent):
            self.parents.append(p)
            p.children.append(self)

    def add_child(self, child):
        """"This layer will send an output tensor to `child`."""
        for c in _make_iterable(child):
            self.children.append(c)
            c.parents.append(self)

    def add_weights(self, w):
        """Add w to this layer's weights."""
        self.weights.extend(_make_iterable(w))

    def __call__(self, parent):
        """This layer will recieve an input tensor from `parent`.

        Syntactic sugar around `add_parent` function.

        """
        self.add_parent(parent)

# Generate Layer sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Layer message in lbann.proto
_generate_classes_from_message(
    Layer, lbann_pb2.Layer,
    skip_fields=set([
        'name', 'parents', 'children', 'data_layout', 'device_allocation',
        'weights', 'num_neurons_from_data_reader', 'freeze', 'hint_layer',
        'weights_data', 'top', 'bottom', 'type', 'motif_layer']),
    base_kwargs=[('parents', []), ('children', []), ('weights', []),
                 ('name', None), ('data_layout', 'data_parallel'),
                 ('hint_layer', None)])

def traverse_layer_graph(layers):
    """Generator function for a topologically ordered graph traversal.

    `layers` should be a `Layer` or a sequence of `Layer`s. All layers
    that are connected to `layers` will be traversed.

    The layer graph is assumed to be acyclic. Strange things may
    happen if this does not hold.

    """

    # DFS to find root nodes in layer graph
    roots = []
    visited = set()
    stack = list(_make_iterable(layers))
    while stack:
        l = stack.pop()
        if l not in visited:
            visited.add(l)
            if not l.parents:
                roots.append(l)
            else:
                stack.extend(l.parents)

    # DFS to traverse layer graph in topological order
    visited = set()
    stack = roots
    while stack:
        l = stack.pop()
        if (l not in visited
            and all([(p in visited) for p in l.parents])):
            visited.add(l)
            stack.extend(l.children)
            yield l

# ==============================================
# Weights and weight initializers
# ==============================================

# Set up weight initializers.
class Initializer:
    """Base class for weight initializers."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Should be overridden in all sub-classes
        raise NotImplementedError

# Generate Initializer sub-classes from lbann.proto.
# Note: The list of skip fields must be updated if any new fields are
# added to the Weights message in lbann.proto
_generate_classes_from_message(
    Initializer, lbann_pb2.Weights,
    skip_fields=set(['name', 'optimizer']),
    type_has_parent=False)

class Weights:
    """Trainable model parameters."""

    global_count = 0  # Static counter, used for default names

    def __init__(self, initializer=None, optimizer=None, name=None):
        Weights.global_count += 1
        self.name = name if name else 'weights{0}'.format(Weights.global_count)
        self.initializer = initializer
        self.optimizer = optimizer

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Weights()
        proto.name = self.name

        # Set initializer if needed
        if self.initializer:
            type_name = type(self.initializer).__name__
            field_name = None
            for field in lbann_pb2.Weights.DESCRIPTOR.fields:
                if field.message_type and field.message_type.name == type_name:
                    field_name = field.name
                    break
            init_message = getattr(proto, field_name)
            init_message.CopyFrom(self.initializer.export_proto())
            init_message.SetInParent()

        # TODO: implement
        if self.optimizer:
            raise NotImplementedError('Weights cannot handle non-default optimizers')

        return proto

# ==============================================
# Objective functions
# ==============================================

# Note: Currently, only layer terms and L2 weight regularization terms
# are supported in LBANN. If more terms are added, it may be
# worthwhile to autogenerate sub-classes of ObjectiveFunctionTerm.

class ObjectiveFunctionTerm:
    """Base class for objective function terms."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Should be overridden in all sub-classes
        raise NotImplementedError

class LayerTerm(ObjectiveFunctionTerm):
    """Objective function term that takes value from a layer."""

    def __init__(self, layer, scale=1.0):
        self.layer = layer
        self.scale = scale

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.LayerTerm()
        proto.layer = self.layer.name
        proto.scale_factor = self.scale
        return proto

class L2WeightRegularization(ObjectiveFunctionTerm):
    """Objective function term for L2 regularization on weights."""

    def __init__(self, weights=[], scale=1.0):
        self.scale = scale
        self.weights = list(_make_iterable(weights))

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.L2WeightRegularization()
        proto.scale_factor = self.scale
        proto.weights = ' '.join([w.name for w in self.weights])
        return proto

class ObjectiveFunction:
    """Objective function for optimization algorithm."""

    def __init__(self, terms=[]):
        """Create an objective function with layer terms and regularization.

        `terms` should be a sequence of `ObjectiveFunctionTerm`s and
        `Layer`s.

        """
        self.terms = []
        for t in _make_iterable(terms):
            self.add_term(t)

    def add_term(self, term):
        """Add a term to the objective function.

        `term` may be a `Layer`, in which case a `LayerTerm` is
        constructed and added to the objective function.

        """
        if isinstance(term, Layer):
            term = LayerTerm(term)
        self.terms.append(term)

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.ObjectiveFunction()
        for term in self.terms:
            term_message = term.export_proto()
            if type(term) is LayerTerm:
                proto.layer_term.extend([term_message])
            elif type(term) is L2WeightRegularization:
                proto.l2_weight_regularization.extend([term_message])
        return proto

# ==============================================
# Metrics
# ==============================================

class Metric:
    """Metric that takes value from a layer.

    Corresponds to a "layer metric" in LBANN. This may need to be
    generalized if any other LBANN metrics are implemented.

    """

    def __init__(self, layer, name=None, unit=''):
        """Initialize a metric based of off layer."""
        self.layer = layer
        self.name = name if name else self.layer.name
        self.unit = unit

    def export_proto(self):
        """Construct and return a protobuf message."""
        proto = lbann_pb2.Metric()
        proto.layer_metric.layer = self.layer.name
        proto.layer_metric.name = self.name
        proto.layer_metric.unit = self.unit
        return proto

# ==============================================
# Callbacks
# ==============================================

class Callback:
    """Base class for callbacks."""

    def __init__(self):
        pass

    def export_proto(self):
        """Construct and return a protobuf message."""
        return lbann_pb2.Callback()

# Generate Callback sub-classes from lbann.proto
# Note: The list of skip fields must be updated if any new fields are
# added to the Callback message in lbann.proto
_generate_classes_from_message(Callback, lbann_pb2.Callback)

# ==============================================
# Model
# ==============================================

class Model:
    """Base class for models."""

    def __init__(self, mini_batch_size, epochs,
                 layers, weights=[], objective_function=None,
                 metrics=[], callbacks=[]):
        self.mini_batch_size = mini_batch_size
        self.epochs = epochs
        self.layers = layers
        self.weights = weights
        self.objective_function = objective_function
        self.metrics = metrics
        self.callbacks = callbacks

    def export_proto(self):
        """Construct and return a protobuf message."""
        # Initialize protobuf message
        model = lbann_pb2.Model()
        model.mini_batch_size = self.mini_batch_size
        model.num_epochs = self.epochs
        model.block_size = 256           # TODO: Make configurable.
        model.num_parallel_readers = 0   # TODO: Make configurable
        model.procs_per_trainer = 0      # TODO: Make configurable

        # Add layers
        layers = list(traverse_layer_graph(self.layers))
        model.layer.extend([l.export_proto() for l in layers])

        # Add weights
        weights = set(self.weights)
        for l in layers:
            weights.update(l.weights)
        model.weights.extend([w.export_proto() for w in weights])

        # Add objective function
        objective_function = self.objective_function \
            if self.objective_function else ObjectiveFunction()
        model.objective_function.CopyFrom(objective_function.export_proto())

        # Add metrics and callbacks
        model.metric.extend([m.export_proto() for m in self.metrics])
        model.callback.extend([c.export_proto() for c in self.callbacks])

        return model

# ==============================================
# Export models
# ==============================================

def save_model(filename, *args, **kwargs):
    """Create a model and save to a file.
    This function delegates all the arguments to `lp.Model` except
    for `filename`.
    """

    save_prototext(filename,
                   model=Model(*args, **kwargs).export_proto())

def save_prototext(filename, **kwargs):
    """Save a prototext.
    This function accepts the LbannPB objects via `kwargs`, such as
    `model`, `data_reader`, and `optimizer`.
    """

    # Initialize protobuf message
    pb = lbann_pb2.LbannPB(**kwargs)

    # Write to file
    with open(filename, 'wb') as f:
        f.write(google.protobuf.text_format.MessageToString(
            pb, use_index_order=True).encode())
