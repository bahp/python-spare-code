"""
script_01.py
=============

Example
"""
import ldap3

from ldap_test import LdapServer

server = LdapServer({
    'port': 3333,
    'bind_dn': 'cn=admin,dc=zoldar,dc=net',
    'password': 'pass1',
    'base': {'objectclass': ['domain'],
             'dn': 'dc=zoldar,dc=net',
             'attributes': {'dc': 'zoldar'}},
    'entries': [
        {'objectclass': 'domain',
         'dn': 'dc=users,dc=zoldar,dc=net',
         'attributes': {'dc': 'users'}},
        {'objectclass': 'organization',
         'dn': 'o=foocompany,dc=users,dc=zoldar,dc=net',
         'attributes': {'o': 'foocompany'}},
    ]
})

try:
    server.start()

    dn = "cn=admin,dc=zoldar,dc=net"
    pw = "pass1"

    srv = ldap3.Server('localhost', port=3333)
    conn = ldap3.Connection(srv, user=dn, password=pw, auto_bind=True)

    base_dn = 'dc=zoldar,dc=net'
    search_filter = '(objectclass=organization)'
    attrs = ['o']

    conn.search(base_dn, search_filter, attributes=attrs)

    print(conn.response)
    # [{
    #    'dn': 'o=foocompany,dc=users,dc=zoldar,dc=net',
    #    'raw_attributes': {'o': [b'foocompany']},
    #    'attributes': {'o': ['foocompany']},
    #    'type': 'searchResEntry'
    # }]
finally:
    server.stop()