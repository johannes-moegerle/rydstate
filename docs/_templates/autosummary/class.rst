{{ name | escape | underline }}

.. currentmodule:: {{ module }}

{% set own_methods = methods | reject("in", inherited_members) | list %}
{% if module.endswith("_model_data") or ((name.startswith("MQDT") or name.startswith("SQDT")) and own_methods | length == 0) %}
.. The species data classes (MQDTModel / MQDT / SQDT subclasses) only store
   parameters (class attributes) for a given species; they inherit all of their
   behaviour (methods and computed properties) from their respective base class.
   Here we only document the parameters that are actually defined on the
   subclass, and hide the inherited members.

.. autoclass:: {{ fullname }}
   :members:
   :member-order: bysource
   :undoc-members:
   :exclude-members: __init__
{% else %}
{% if methods %}
.. rubric:: Class Methods

.. autosummary::
   {% for item in methods %}
      ~{{ name }}.{{ item }}
   {%- endfor %}
{% endif %}

{% if attributes %}
.. rubric:: Class Attributes and Properties

.. autosummary::
{% for item in attributes %}
   ~{{ name }}.{{ item }}
{%- endfor %}
{% endif %}


.. autoclass:: {{ fullname }}
   :members:
   :member-order: bysource
   :inherited-members:
   :undoc-members:
   :class-doc-from: init
{% endif %}
